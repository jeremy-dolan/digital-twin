#!/usr/bin/env python3
import urllib.request

url = "https://jeremy-dolan-digital-twin.hf.space"
resp = urllib.request.urlopen(url)
print(f"{resp.status} {url}")
