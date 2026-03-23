#!/usr/bin/env python3
"""Test whether the Gradio /call/ API endpoint is accessible"""
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:7860"
# DEFAULT_URL = "https://jeremy-dolan-digital-twin.hf.space"
ENDPOINT = "/gradio_api/call/gradio_input_callback"
PAYLOAD = {"data": ["Hello, what's your name?", [], []]}


base_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else DEFAULT_URL
submit_url = base_url + ENDPOINT


### Submit request
print(f"POST {submit_url}")
req = urllib.request.Request(
    submit_url,
    data=json.dumps(PAYLOAD).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    resp = urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    sys.exit(f"  → {e.code} {e.reason}")
except urllib.error.URLError as e:
    sys.exit(f"  → Connection failed: {e.reason}")

body = json.loads(resp.read())
event_id = body.get("event_id")
if not event_id:
    sys.exit(f"  → Unexpected response: {body}")
print(f"  → event_id: {event_id}")


### Stream result
result_url = f"{submit_url}/{event_id}"
print(f"GET  {result_url}")
resp = urllib.request.urlopen(result_url)
for line in resp:
    line = line.decode().strip()
    if line.startswith("data:"):
        data = json.loads(line[len("data:"):].strip())
        # Response is in data[0] (the chatbot messages list)
        if data and data[0]:
            for msg in data[0]:
                content = msg.get("content", "")
                if content:
                    print(f"  → LLM response: {content}")
    if line.startswith("event: complete"):
        break
