import os

import chromadb
import gradio as gr
from openai import OpenAI
from openai.types.responses import ResponseInputItemParam

import config
import inference
import prompts
import rag
import tools


### setup

on_hf_spaces = os.environ.get("SPACE_ID") is not None
if on_hf_spaces:
    # On Hugging Face Spaces, first download biography vector store from dataset repo
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=config.HUGGINGFACE_DATASET_REPO,
        repo_type='dataset',
        local_dir=config.CHROMA_PATH.name,
    )
else:
    # If local, vector store should already be built and available at config.CHROMA_PATH
    from dotenv import load_dotenv
    load_dotenv()


oai_client = OpenAI()

chroma_client = chromadb.PersistentClient(config.CHROMA_PATH, config.CHROMA_CLIENT_SETTINGS)
collection = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)

tool_registry = tools.build_all_tools()


### conversation state
#
# Gradio's ChatInterface maintains a per-session message history and passes it to the callback
# function (gradio_loop) on each turn. This means that we cannot manually retain additional
# intra-turn messages (tool calls and responses, context injections).
# 
# To maintain additional per-session state: https://www.gradio.app/guides/interface-state
#     TODO: I should maintain injected chunk messages for consistent dialogue
# To capture UI interactions with state: https://www.gradio.app/guides/chatbot-specific-events

def gradio_to_oai_history(gradio_history: list[dict]) -> list[ResponseInputItemParam]:
    """
    Normalize the history we recieve from Gradio's ChatInterface before sending to OpenAI.
    Gradio API docs say that message history is a "list of openai-style dictionaries," but it
    includes additional keys (notably, a 'metadata' key) that trigger a "BadRequestError" from
    OpenAI's Responses API.
    """
    normalized_history: list[ResponseInputItemParam] = []
    for item in gradio_history:
        if not ('role' in item and 'content' in item):
            print(f"Unexpected format for history item: {item}")
            continue
        if (item.get('metadata') or {}).get('title'):
            # if metadata is populated, message is a thought accordian; skip it
            continue
        if item['role'] not in ['user', 'assistant', 'developer']:
            print(f"Unexpected role in history item: {item['role']}")
            continue
        normalized_history.append({'role': item['role'], 'content': item['content'][0]['text']})
    return normalized_history


def gradio_input_callback(input: str, gradio_history: list[dict]):
    """
    Called when the user inputs a new message. Handle a single conversation turn:
    retrieve context, call LLM, stream response with reasoning/tool metadata.
    """
    # TODO: Still unclear after much research whether context injection should use role=user or
    # role=developer. Could use some empirical testing.

    rag_context = rag.build_context_injection(oai_client, collection, input)

    messages: list[ResponseInputItemParam] = []
    messages.append({"role": "developer", "content": prompts.SYSTEM_MESSAGE})
    messages.extend(gradio_to_oai_history(gradio_history[1:]))     # skip greeting, normalize rest
    messages.append({"role": "developer", "content": rag_context}) # role=user or role=developer?
    messages.append({"role": "user", "content": input})

    print("---about to call stream_turn for---")
    for m in messages[1:]:
        print(m)
    print("------------------------------------")

    # could insert a message saying... Retrieved [x] memories. Ran Y and Z tools.
    # for tool use (and citations) display: https://www.gradio.app/guides/agents-and-tool-usage
    yield from inference.stream_turn(oai_client, messages, tool_registry)


### Gradio UI

greeting: gr.MessageDict = {
    "role": "assistant", "content": "Hi! 👋 I'm Virtual Jeremy. How can I help?"
}
chatbot = gr.Chatbot(
    [greeting],
    editable="user",  # allow users to edit user messages (but not assistant messages)
    show_label=False, # declutter (top-left within chat box); label="" to set
    avatar_images=(None, config.BASE_DIR / 'assets' / 'avatar.png'),
    buttons=['copy_all', 'share', 'copy'],
    # placeholder='text centered in chatbot box', # not shown due to greeting
    scale=1,
)
demo = gr.ChatInterface(
    fn=gradio_input_callback,
    chatbot=chatbot,
    editable=True,
    title='Virtual Jeremy', # HTML title *and* <h1> text above the chatbot
    # description="Jeremy Dolan's digital twin. Built with Gradio, OpenAI, and ChromaDB.",
    # examples=['What have you been up to lately?'], # not shown due to greeting
    api_visibility="private",
    analytics_enabled=False,
    fill_height=True, # Was broken in Gradio 6.8, but is fixed in 6.9. (PR#12956)
                      # Workaround was to pass elem_id="chatbot" to gr.Chatbot(), then add
                      # to custom_css: "#chatbot { height: calc(100vh - 150px) !important; }"
    fill_width=True,
)

custom_css = (
    # Hugging Face left-aligns the title on its own; make local do the same:
    "h1 { text-align: left !important; }\n"

    # Make the avatars bigger: increase container size, remove padding (within the circle), and
    #   vertically center chat bubbles to improve rendering for single-line assistant messages
    ".avatar-container { width: 50px !important; height: 50px !important; }\n"
    ".avatar-container img { padding: 0 !important; }\n"
    ".role { align-self: center !important; }\n"
    # ".avatar-container { align-self: center !important; }\n" # center align the avatars
    # ".avatar-container { align-self: flex-end !important; }\n" # bottom align the avatars

    # ad hoc patch to make buttons still align after increasing avatar size.
    ".message-buttons-left { margin-left: calc(var(--spacing-xl) * 6.5) !important; }\n"
 
    # Workaround for Gradio iframe resizer bug on HF Spaces.
    # footer_links=[] removes the footer from the DOM, causing infinite vertical growth
    # Hiding via CSS keeps the element in the DOM as an anchor for the iframe height calculation.
    # "footer { display: none !important; }\n" DOESN'T WORK
    "footer { height: 5px !important; visibility: hidden !important; }\n"
    # TODO: TRY NEW LOGO
)

if __name__ == "__main__":
    demo.launch(
        footer_links=['settings'], # if empty, HF Spaces iframe sizing bug; CSS workaround is above
        favicon_path=config.BASE_DIR / 'assets' / 'favicon.ico',
        theme="origin",
        css=custom_css
    )
