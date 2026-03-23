import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox", api_visibility="private")
demo.launch()
