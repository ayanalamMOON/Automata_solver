import os
from openai import OpenAI

# Get API key from environment variable - safer than hardcoding
api_key = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key)
