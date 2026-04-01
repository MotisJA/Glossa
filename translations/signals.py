from django.db.models.signals import post_delete, post_save
from django.db import transaction
from django.dispatch import receiver

from .models import GlossaryEntry


@receiver(post_save, sender=GlossaryEntry)
def invalidate_glossary_trie_on_save(sender, **kwargs):
    transaction.on_commit(GlossaryEntry.invalidate_trie_cache)


@receiver(post_delete, sender=GlossaryEntry)
def invalidate_glossary_trie_on_delete(sender, **kwargs):
    transaction.on_commit(GlossaryEntry.invalidate_trie_cache)
