import os
import json
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import dspy

from .models import GlossaryEntry, SystemConfiguration, CorpusEntry

lm = dspy.LM('gemini/gemini-2.0-flash', api_key=os.getenv('GEMINI_API_KEY'))
dspy.configure(lm=lm)

config = SystemConfiguration.load()

class GlossaryEntrySimplified(BaseModel):
    en: str = Field(title="The English key")
    tgt: str = Field(title="The translated entry")
    target_language: str = Field(title="Target language code", default="zh")

class CorpusEntrySimplified(BaseModel):
    en: str = Field(title="The English text")
    tgt: str = Field(title="The translated text")
    target_language: str = Field(title="Target language code", default="zh")
    machine_translation: str = Field(title="The machine translation of the English text", default="")


class Input(BaseModel):
    input_text: str = Field(title="The text to translate to Bislama")
    machine_translated: str = Field(title="A first-pass machine translation of the text")
    glossary_entries: List[GlossaryEntrySimplified] = Field(
        default_factory=list,
        description="Candidate glossary entries. First check if each one matches the same domain and target language; then decide whether to use it.",
    )
    past_translations: List[CorpusEntrySimplified] = Field(
        default_factory=list,
        description="Candidate examples. First check whether each example is in the same domain and target language before deciding to use it.",
    )

    @classmethod
    async def from_english_text(cls, english_text: str, translator):
        glossary_entries = GlossaryEntry.get_entries(english_text, target_language=config.target_language_code)
        glossary_entries = [ge.as_dict() for ge in glossary_entries]
        machine_translated = await translator.translate(english_text)
        similar_sentences = CorpusEntry.get_top_similar_bm25(
            english_text,
            config.num_sentences_retrieved,
            target_language=config.target_language_code,
        )
        similar_sentences = [l.as_dict() for l in similar_sentences]
        return cls(input_text=english_text, machine_translated=machine_translated, glossary_entries=glossary_entries, past_translations=similar_sentences)

class Output(BaseModel):
    output_text: str = Field(title="The translated text")

class PostEditSignature(dspy.Signature):
    """Before applying glossary/examples, evaluate if they match the source domain and target language; only use those that help."""

    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()
