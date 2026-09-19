import os, requests, json
from dotenv import load_dotenv
load_dotenv()
ENDPOINT = os.environ["ARVAN_ENDPOINT"]
API_KEY = os.environ["ARVAN_API_KEY"]

# Replace with whatever exact model string Arvan uses for gpt-4o-mini in
# your dashboard -- likely something like "gpt-4o-mini" or a longer suffixed
# name, mirroring how your Llama model is "Llama-3.3-70B-Instruct-R1AI".
MODEL = "gpt-4o-mini"

resp = requests.post(
    f"{ENDPOINT}/chat/completions",
    headers={"Authorization": f"apikey {API_KEY}", "Content-Type": "application/json"},
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with only the letter B."}],
        "max_tokens": 5,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    },
    timeout=(30, 60),
)

print("HTTP status:", resp.status_code)
data = resp.json()
print(json.dumps(data, indent=2)[:3000])

choice = data.get("choices", [{}])[0]
has_logprobs = choice.get("logprobs") is not None
print("\n--- RESULT ---")
print("logprobs field present and non-null:", has_logprobs)
if has_logprobs:
    print("content entries:", len(choice["logprobs"].get("content", [])))