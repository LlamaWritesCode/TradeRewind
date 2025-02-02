import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("API_KEY")

if not OPENAI_KEY:
    raise ValueError("API_KEY is missing. Check your .env file.")
