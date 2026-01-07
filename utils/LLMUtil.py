from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in environment (.env)")

api = OpenAI(api_key=api_key, base_url=base_url)


def ask_MrGPT(
    prompt: str,
    system_prompt: str = None,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    completion = api.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=8000,
    )

    return completion.choices[0].message.content
