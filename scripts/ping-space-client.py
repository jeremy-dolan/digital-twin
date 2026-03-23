#!/usr/bin/env python3
from gradio_client import Client

client = Client("jeremy-dolan/digital-twin")
# client = Client('http://127.0.0.1:7860')
payload=["what's up", [], []]
# should fail due to *client side* check for visibility==hidden
result = client.predict(payload, api_name="/gradio_input_callback")
print(result)
