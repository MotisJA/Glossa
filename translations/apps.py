from django.apps import AppConfig


class TranslationsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'translations'

    def ready(self):
        # Ensure signal handlers are registered on app startup.
        from . import signals  # noqa: F401
