from dotenv import load_dotenv

import os

load_dotenv()

print(os.environ.get("OPENAI_API_KEY"))