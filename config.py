import os

AZURE_OPENAI_ENDPOINT = "https://ai-proxy.lab.epam.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-1106-preview"
OPENAI_API_VERSION = "2023-07-01-preview"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

TEMPERATURE = 0.5
MAX_TOKENS = 200