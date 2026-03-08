---
title: Virtual Jeremy
short_description: Chat with Jeremy Dolan's Digital Twin
thumbnail: https://jeremydolan.net/digital-twin/assets/logo.png
emoji: 🤖
colorFrom: blue
colorTo: indigo
python_versino: 3.13.12
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
header: mini
startup_duration_timeout: 5m
pinned: true
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

A RAG-powered chatbot that responds as a digital version of Jeremy Dolan. 

<!--
Built using:
 - [Gradio](https://gradio.app)
 - [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index)
 - some other stuff

<!-- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index) -->
