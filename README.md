# Glossa
This project is for adaptive and interactive machine translation based on tulun.

## Local installation

1. Install Python dependencies:
```bash
pip install poetry
eval $(poetry env activate)
poetry install
```

2. Setup credentials: create a .env file in the root directory and add the following:
```bash
# for Google Translate (optional)
GOOGLE_APPLICATION_CREDENTIALS='<credential-file>.json'
# for Gemini, can also use OpenAI / Anthropic / others, see https://docs.litellm.ai/docs/
GEMINI_API_KEY='<api-key>'
```

3. Run the server:
```bash
eval $(poetry env activate)
./manage.py migrate && ./manage.py runserver
```