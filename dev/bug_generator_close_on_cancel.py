"""
Minimal repro for: Gradio ChatInterface doesn't close sync generators on cancel
Reported as gradio-app/gradio#13044

Run this, send a long message, then click the stop button while it's streaming.

## Expected

GeneratorExit is raised, "GENERATOR CLOSED" is printed.
App can shut down request gracefully/close API streams.

## Actual

Error is printed to console:
```
gradio/utils.py:1942: RuntimeWarning: coroutine method 'aclose' of 'slow_echo' was never awaited
  iterator.aclose()
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
```
Then generator runs one additional iteration and silently dies.
"""

import time
import gradio as gr

def slow_echo(message, history):
    try:
        for i, char in enumerate(message):
            time.sleep(0.5)
            print(f"yield {i}: {char!r}")
            yield char
        print("GENERATOR EXHAUSTED NATURALLY")
    except GeneratorExit:
        print("GENERATOR CLOSED")
    finally:
        print("FINALLY BLOCK")


gr.ChatInterface(slow_echo).launch()
