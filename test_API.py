import os, json
from dotenv import load_dotenv
load_dotenv()

import openai

openai.api_key = os.environ["CHABOK_API_KEY"]      # your Chabokan key
openai.api_base = "https://ai.chabokan.net/v1"

resp = openai.ChatCompletion.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Reply with only the letter B."}],
    max_tokens=5,
    temperature=0.0,
    logprobs=True,        # <-- this is the actual test, missing from your snippet
    top_logprobs=5,
)

print(json.dumps(resp, indent=2)[:3000] if isinstance(resp, dict) else resp)

choice = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
logprobs_field = choice["logprobs"] if isinstance(choice, dict) else choice.logprobs

print("\n--- RESULT ---")
print("logprobs field present and non-null:", logprobs_field is not None)
if logprobs_field:
    content = logprobs_field["content"] if isinstance(logprobs_field, dict) else logprobs_field.content
    print("content entries:", len(content))
    print("first entry:", content[0])