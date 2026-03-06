import os

import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputItemParam

import config
import inference
import prompts
import rag
import tools


### setup

load_dotenv()

oai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)
tool_registry = tools.build_all_tools()
tool_names = list(tool_registry.tools.keys()) # XXX: do we want to pre-compute this?


### conversation state
#
# Gradio's ChatInterface maintains a per-session message history and passes it to the callback
# function (gradio_loop) on each turn. This means that we cannot manually retain additional
# intra-turn messages (tool calls and responses, context injections).
# 
# To maintain additional per-session state: https://www.gradio.app/guides/interface-state
# To capture UI interactions with state: https://www.gradio.app/guides/chatbot-specific-events

def gradio_to_oai_history(gradio_history: list[dict]) -> list[ResponseInputItemParam]:
    """
    Normalize the history we recieve from Gradio's ChatInterface before sending to OpenAI.
    Gradio API docs say that message history is a "list of openai-style dictionaries,"
    but in practice it includes additional keys (notably, a 'metadata' key) that cause
    OpenAI's Responses API to reject the request with "BadRequestError."
    """
    normalized_history: list[ResponseInputItemParam] = []
    for item in gradio_history:
        if not ('role' in item and 'content' in item):
            print(f"Unexpected history item format: {item}")
            continue
        if item['role'] not in ['user', 'assistant', 'developer']:
            print(f"Unexpected role in history item: {item['role']}")
            continue
        normalized_history.append({'role': item['role'], 'content': item['content'][0]['text']})
    return normalized_history


def gradio_msg_input_callback(input: str, gradio_history: list[dict]) -> str:
    """
    Called on each user message to handle a single conversation turn:
    retrieve context, call LLM, return response text.
    """
    # TODO: Still unclear after much research whether context injection should use role=user or
    # role=developer. Could use some empirical testing.

    rag_context = rag.build_context_injection(oai_client, collection, input)

    messages: list[ResponseInputItemParam] = []
    messages.append({"role": "developer", "content": prompts.SYSTEM_MESSAGE})
    messages.extend(gradio_to_oai_history(gradio_history[1:]))     # skip greeting, normalize rest
    messages.append({"role": "developer", "content": rag_context}) # role=user or role=developer?
    messages.append({"role": "user", "content": input})

    print("----")
    for m in messages[1:]:
        print(m)
    print("----")

    # could insert a message saying... Retrieved [x] memories. Ran Y and Z tools.
    # for tool use (and citations) display: https://www.gradio.app/guides/agents-and-tool-usage
    model_response_text = inference.resolve_turn(oai_client, messages, tool_registry)

    return model_response_text


### Gradio UI

greeting: gr.MessageDict = {
    "role": "assistant", "content": "🤖👋 Hi, I'm Virtual Jeremy. Ask me anything."
}
chatbot = gr.Chatbot(
    [greeting],
    show_label=False, # declutter (top-left within chat box); label="" to set
    avatar_images=(None, config.BASE_DIR / 'assets' / 'avatar.jpg'),
    buttons=['copy_all', 'share', 'copy'],
    scale=1,
    elem_id="chatbot",
)
demo = gr.ChatInterface(
    fn=gradio_msg_input_callback,
    chatbot=chatbot,
    api_visibility="private",
    fill_height=True, # appears broken in Gradio 6.8; CSS workaround in launch() below
    fill_width=True,
    # description='Virtual Jeremy', # page header
)

if __name__ == "__main__":
    demo.launch(
        footer_links=[],
        favicon_path=config.BASE_DIR / 'assets' / 'favicon.ico',
        theme="origin",
        css="#chatbot { height: calc(100vh - 150px) !important; }", # fill_height workaround
    )
