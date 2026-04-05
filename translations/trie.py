from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import GlossaryEntry


@dataclass
class TrieNode:
    """A node in the glossary trie.

    The node tracks its child characters, whether this node marks the end of a
    term, and matched glossary entries for the term.
    """

    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end_of_term: bool = False
    entry_ids: List[int] = field(default_factory=list)


class Trie:
    """Character-level trie for case-insensitive glossary lookup."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, term: str, entry_obj: "GlossaryEntry") -> None:
        normalized = term.strip().lower()
        if not normalized:
            return

        node = self.root
        for char in normalized:
            node = node.children.setdefault(char, TrieNode())

        node.is_end_of_term = True
        if entry_obj.pk is None:
            return
        node.entry_ids.append(int(entry_obj.pk))

    def build(self, queryset: Iterable["GlossaryEntry"]) -> None:
        for entry in queryset:
            self.insert(entry.english_key, entry)

    @staticmethod
    def _is_boundary(text: str, start: int, end: int) -> bool:
        has_left_boundary = start == 0 or not text[start - 1].isalnum()
        has_right_boundary = end == len(text) - 1 or not text[end + 1].isalnum()
        return has_left_boundary and has_right_boundary

    def extract_longest_match_ids(self, text: str) -> List[int]:
        """Extract terms using Forward Maximum Matching (FMM).

        The scan is case-insensitive and prefers the longest valid term for each
        start position. English term boundaries are enforced to avoid partial word
        matches like extracting "cat" from "category".
        """

        normalized_text = text.lower()
        matches: List[int] = []
        seen_entry_ids = set()

        start = 0
        text_len = len(normalized_text)

        while start < text_len:
            if start != 0 and text[start-1].isalnum():
                start += 1
                continue

            node = self.root
            current = start
            latest_match: Optional[Tuple[int, TrieNode]] = None

            while current < text_len:
                char = normalized_text[current]
                if char not in node.children:
                    break

                node = node.children[char]
                if node.is_end_of_term:
                    latest_match = (current, node)
                current += 1

            if latest_match is None:
                start += 1
                continue

            match_end, match_node = latest_match
            if not self._is_boundary(normalized_text, start, match_end):
                start += 1
                continue

            for entry_id in match_node.entry_ids:
                if entry_id not in seen_entry_ids:
                    matches.append(entry_id)
                    seen_entry_ids.add(entry_id)

            start = match_end + 1

        return matches
