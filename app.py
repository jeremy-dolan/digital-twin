import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # quiet third-party libraries
for name in (__name__, 'inference', 'rag', 'tools'):
    logging.getLogger(name).setLevel(config.LOG_LEVEL)


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


# SESSION STATE
# 
# Gradio's ChatInterface manages per-session state. Normally the state is just a list of Gradio
# ChatMessages passed to the callback ("gradio_history"), intended to be used both for API
# completions and the UI state. To improve coherence of model responses, I want to maintian a
# more complete message history (including RAG context injections, reasoning traces, and tool
# calls/responses). We add an additional per-state gr.State component ("api_messages") to hold
# the accumulated API messages across turns. (https://www.gradio.app/guides/interface-state)
# Gradio's own chat history is used only for display rendering, and is not accessed here.

def gradio_input_callback(user_input: str,
                          gradio_history: list[gr.ChatMessage],
                          api_messages: list[ResponseInputItemParam]):
    """
    Called when the user inputs a new message. Manages `api_messages` (prompt,
    context injection) and hands off to steam_turn for LLM response and tool
    handling. Yields ChatMessages for streaming updates to the UI.
    """
    if not api_messages:
        api_messages.append({"role": "developer", "content": prompts.SYSTEM_MESSAGE})

    rag_context = rag.build_context_injection(oai_client, collection, user_input)
    # TODO: Still unclear after much research whether context injection should use role=user or
    # role=developer. Could use some empirical testing. Injection should likely go before the user
    # query to keep response focused on the most recent message, but that could use testing, too!
    api_messages.append({"role": "developer", "content": rag_context})
    api_messages.append({"role": "user", "content": user_input})

    # Debug messages
    logger.debug("---about to run stream_turn with API messages:---")
    logger.debug("%s", {**api_messages[0], 'content': api_messages[0]['content'][:40] + '...'})
    for m in api_messages[1:]:
        logger.debug("%s", m)
    logger.debug("-------------------------------------------------")

    # TODO: pass len(chunks) to allow '⧉ Retrieved [x] memories' RAG feedback in ThoughtAccordion?
    yield from inference.stream_turn(oai_client, api_messages, tool_registry)


### Gradio UI

greeting: gr.MessageDict = {
    "role": "assistant", "content": "Hi! 👋 I'm Virtual Jeremy. How can I help?"
}
chatbot = gr.Chatbot(
    [greeting],       # initial message history; removed before passing to LLM, purely aesthetic
    # editable="user",  # allow users to edit user messages (but not assistant messages)
    show_label=False, # declutter (top-left within chat box); label="" to set
    avatar_images=(None, config.BASE_DIR / 'assets' / 'avatar.png'),
    # buttons=['copy_all', 'share', 'copy'],
    buttons=[],
    # placeholder='text centered in chatbot box', # not shown due to greeting
    scale=1,
)
api_messages = gr.State([])  # additional Gradio component for per-session state
demo = gr.ChatInterface(
    fn=gradio_input_callback,
    chatbot=chatbot,
    additional_inputs=[api_messages],   # read this component's value and pass to the callback
    additional_outputs=[api_messages],  # store what the callback yields here
    additional_inputs_accordion=gr.Accordion(visible=False),
    # editable=True,
    title='Virtual Jeremy', # HTML title *and* <h1> text above the chatbot
    # description="Jeremy Dolan's digital twin. Built with Gradio, OpenAI, and ChromaDB.",
    # examples=['What have you been up to lately?'], # not shown due to greeting
    api_visibility="private",
    analytics_enabled=False,
    fill_height=True, # Was broken in Gradio 6.8, but is fixed in 6.9. (PR#12956)
                      # Workaround was to pass elem_id="chatbot" to gr.Chatbot(), then add
                      # to custom_css: "#chatbot { height: calc(100vh - 150px) !important; }"
    fill_width=False,
)

custom_css = (
    ".main { max-width: 800px !important; margin: auto !important; }\n"

    # Hugging Face left-aligns the title; make local do the same:
    "h1 { text-align: left !important; }\n"

    # Make the avatars bigger: increase container size, remove padding within the circle
    ".avatar-container { width: 50px !important; height: 50px !important; }\n"
    ".avatar-container img { padding: 0 !important; }\n"

    # vertically center chat bubbles to improve rendering for single-line assistant messages
    ".role { align-self: center !important; }\n"
    # ".avatar-container { align-self: center !important; }\n" # center align the avatars
    # ".avatar-container { align-self: flex-end !important; }\n" # bottom align the avatars

    # ad hoc patch to make buttons horizontally align after increasing avatar size.
    # ".message-buttons-left { margin-left: calc(var(--spacing-xl) * 6.5) !important; }\n"

    # new minimal aesthetic with no buttons or editing
    ".message-buttons-left { display: none !important; }\n"
    ".thought-group { width: fit-content !important; padding-right: var(--spacing-xxl) !important}\n"

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
