import os
import requests
os.environ["ARVAN_API_KEY"] = "d60b8d53-5f6c-5eb6-828a-031d503221d7"
os.environ["ARVAN_ENDPOINT"] = "https://arvancloudai.ir/gateway/models/Llama-3.3-70B-Instruct-R1AI/wXwicG2w6RhESLlIg7KlsetpzCShAX1k0bXHGZLX9nDHLxr4mJwXDMW-2Vunju0LPAXGRBNA8Hg4NhbIgiU-6UwAscgx9QDw1vINQTl1636Buf89X2lEaj3jhmNVJ9Q5qjRgH_G-9CxWta0kJqLV_LtR3XQypA0QnGRFDXtMfKtcE4Q0w5zta-2-Ul-b1rMjyoCaLU2oe4h7NM8-23TJZsUjYp9JrcetdbTBdzh2TDjNK1Iy0I10_c_8RQIN0B9LAEWZihx5ela0-k8AEHGCTwvU5zqzRXjNpwAdEQ/v1"
api_key = os.environ["ARVAN_API_KEY"]
endpoint = os.environ["ARVAN_ENDPOINT"]

print("Testing:", endpoint)

try:
    response = requests.post(
        f"{endpoint}/chat/completions",
        headers={
            "Authorization": f"apikey {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "Llama-3-3-70B-Instruct-R1AI",
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello in one sentence."
                }
            ],
            "max_tokens": 50,
            "temperature": 0.0
        },
        timeout=(10, 60)
    )

    print("Status:", response.status_code)
    print(response.text)

except requests.exceptions.ConnectTimeout:
    print("Could not connect to the endpoint.")
except requests.exceptions.ConnectionError as e:
    print("Connection error:", e)
except Exception as e:
    print(type(e).__name__, e)