import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import chromadb
import gradio as gr
from openai import OpenAI
from openai.types.responses import ResponseInputItemParam
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

import config
import inference
import prompts
import rag
import tools

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,  # quiet third-party libraries
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for name in (__name__, 'inference', 'rag', 'tools', 'request'):
    logging.getLogger(name).setLevel(config.LOG_LEVEL)


### setup

on_hf_spaces = os.environ.get("SPACE_ID") is not None
if on_hf_spaces:
    # on Spaces, first download biography vector store from private dataset repo
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=config.HUGGINGFACE_DATASET_REPO,
        repo_type='dataset',
        local_dir=config.CHROMA_PATH.name,
    )
else:
    # if local, vector store should already be built and available at config.CHROMA_PATH
    from dotenv import load_dotenv
    load_dotenv()


oai_client = OpenAI()
inference.start_summary_probe_loop(oai_client)

chroma_client = chromadb.PersistentClient(config.CHROMA_PATH, config.CHROMA_CLIENT_SETTINGS)
collection = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)

tool_registry = tools.build_all_tools()


### NOTE ON SESSION MANAGEMENT
# Gradio's ChatInterface automatically manages per-session state. By default the state is a list
# of Gradio ChatMessages passed to the callback ("gradio_history"), intended to be used both for
# API completions and the UI state. To improve coherence of model responses, we add an additional
# per-session gr.State component ("api_messages") to hold all accumulated API messages (including
# RAG context injections, reasoning traces, and tool calls/responses) across turns. Gradio's own
# history is only used for display rendering, and is not accessed here -- we just yield updates.
# NOTE for telemetry: add ``request: gr.Request`` to params, then session_id = request.session_hash
def gradio_input_callback(user_input: str,
                          gradio_history: list[gr.ChatMessage],
                          api_messages: list[ResponseInputItemParam]):
    """
    Called when the user inputs a new message. Manages `api_messages` (prompt setup,
    context injection) and hands off to stream_turn for LLM reasoning, tool handling,
    and response. Yields a tuple back to Gradio's session management: ChatMessages
    for streaming updates to the UI, and the entire session's `api_messages`.
    """
    if not user_input.strip():
        # skip empty inputs (server-side backup, JS hack below should catch client-side)
        yield [], api_messages
        return

    if not api_messages:
        api_messages.append({"role": "developer", "content": prompts.SYSTEM_MESSAGE})

    ### RAG
    # add a RAG accordion message to show retrieval in progress
    rag_accordion = gr.ChatMessage(
        role="assistant",
        content="",
        metadata={"title": "🤔 Remembering...", "status": "pending"},
    )
    new_ui_msgs = [rag_accordion]
    yield new_ui_msgs, api_messages

    rag_context, n_chunks, sections = rag.retrieve_context(oai_client, collection, user_input)

    # finalize RAG accordion
    if n_chunks:
        rag_accordion.metadata = {
            "title": f"🤔 Recalled **{n_chunks}** {'memory' if n_chunks == 1 else 'memories'}",
            "status": "done"
            }
        yield new_ui_msgs, api_messages  # close accordion before .content grows its width
        rag_accordion.content = f"Remembered Jeremy's {'; '.join(
            s if s.startswith('AI') else s.lower() for s in sections)}"
        # maybe try as a (non-lowered) bulleted list, since it's hidden anyway?
    else:
        rag_accordion.metadata = {'title': '🤔 No memories found', 'status': 'done'}
    yield new_ui_msgs, api_messages

    ### LLM response
    # TODO: Still unclear after much research whether context injection should use role=user or
    # role=developer. Could use some empirical testing. Injection should likely go before the user
    # query to keep response focused on the most recent message, but that could use testing, too!
    api_messages.append({"role": "developer", "content": rag_context})
    api_messages.append({"role": "user", "content": user_input})

    yield from inference.stream_turn(oai_client, tool_registry, new_ui_msgs, api_messages)


def thematic_motd():
    now = datetime.now(ZoneInfo("America/New_York"))
    month, day, weekday = now.month, now.day, now.weekday()

    if month == 1 and day == 1:
        greet = "Happy New Year! 🎉 I'm Virtual Jeremy. How can I help?"
    elif month == 2 and day == 29:
        greet = "🦘 Happy leap day! A reminder of the distinction between social conventions " \
                "and celestial mechanics. A hypothetical day. A repair to the order we have " \
                "imposed upon reality. Anyway, hi! 👋 I'm Virtual Jeremy. How can I help?"
    elif month == 3 and day == 14:
        greet = "🥧 Happy Pi Day! I'm Virtual Jeremy. How can I help?"
    elif month == 3 and day == 17:
        greet = "☘️ Sláinte! Happy St. Patrick's! I'm Virtual Jeremy. How can I help?"
    elif month == 5 and day == 4:
        greet = "Hi! 👋 I'm Virtual Jeremy. Happy Star Wars Day. May the force be with you!"
    elif month == 4 and day == 1:
        greet = "Hi! 👋 This is the real Jeremy... April Fools. I'm Virtual Jeremy. How can I help?"
    elif month == 10 and day == 31:
        greet = "🎃 Happy Halloween! (The best holiday!) I'm Virtual Jeremy. How can I help?"
    elif month == 12 and (day == 24 or day == 25):
        greet = "🎅🏼 Ho ho ho — I'm Virtual Santa! Just kidding, I'm Virtual Jeremy. How can I help?"
    elif month == 12 and day == 31:
        greet = "Happy New Year's Eve! 🥂 I'm Virtual Jeremy. How can I help?"
    elif weekday == 4:  # Friday
        greet = "Happy Friday! 👋 I'm Virtual Jeremy. How can I help?"
    else:
        greet = "Hi! 👋 I'm Virtual Jeremy. How can I help?"

    return greet


### Gradio UI

greeting: gr.MessageDict = { "role": "assistant", "content": thematic_motd() }
chatbot = gr.Chatbot(
    [greeting],         # starting messages to display (for UI only)
    show_label=False,   # declutter (top-left text within chat box)
    buttons=[],         # declutter
    avatar_images=(None, config.BASE_DIR / 'assets' / 'avatar.png'),
    # editable="user",  # allow users to edit user messages (but not assistant messages)
    # placeholder='text centered in chatbot box', # not shown due to greeting
    scale=1,
)
api_messages = gr.State([])             # additional per-session state
demo = gr.ChatInterface(
    fn=gradio_input_callback,
    chatbot=chatbot,
    additional_inputs=[api_messages],   # read this component's value and pass to the callback
    additional_outputs=[api_messages],  # store additional callback yields here
    additional_inputs_accordion=gr.Accordion(visible=False),  # don't display component in the UI
    title='Virtual Jeremy',             # HTML title *and* <h1> text above the chatbot
    show_progress='hidden',             # spinner conflicts with early display of RAG accordion
    # editable=True,                    # fixed by my PR#12997; but now complicated by api_messages
    # description="Jeremy Dolan's digital twin. Built with Gradio, OpenAI, and ChromaDB.",
    # examples=['What have you been up to lately?'],  # not shown due to greeting
        # for similar functionality add options=[{"label": "L", "value": "input"}, ...] to greeting
    # validator=lambda msg: gr.validate(bool(msg.strip()), "empty"),  # prevent empty input submit
        # failed validation shows label and error message; would take a lot of CSS to make usable
    api_visibility="private",           # NB: endpoint is still usable! (gradio-app/gradio#13051)
    fill_height=True,                   # fixed in 6.9.0 (gradio-app/gradio#12956)
    fill_width=False,
    analytics_enabled=False,
)


with demo:
    # Attach a 'clear' event handler to reset `api_messages` to []
    chatbot.clear(fn=lambda: [], outputs=[api_messages], api_visibility="private")
    # Unfortunately, cannot properly restore `greeting` to the chat box due to how Gradio owns it:
    # XXX chatbot.clear(lambda: ([greeting], []), outputs=[chatbot, api_messages])

    # Hack to block empty submissions client-side
    # Gradio adds the user message to chat history before the server-side callback runs, so a
    # server-side guard can't prevent an empty message bubble from appearing in the UI. Here we
    # capture click/Enter before Gradio sees them. Requires demo.load (not demo.launch(js=...))
    # because Gradio's own event handlers must be attached first for stopImmediatePropagation to
    # intercept them. The trade-off is a spurious server round-trip on page load.
    demo.load(fn=lambda: None, api_visibility="private", js="""() => {
        const isEmpty = () => {
            const tb = document.querySelector('textarea[data-testid="textbox"]');
            return !tb || !tb.value.trim();
        };
        const block = (e) => {
            if (isEmpty()) { e.stopImmediatePropagation(); e.preventDefault(); }
        };
        document.addEventListener('click', e => {
            if (e.target.closest('button.submit-button')) block(e);
        }, true);
        document.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) block(e);
        }, true);
    }""")

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

    # Workaround for Gradio 6.9.0 iframe resizer bug on HF Spaces (gradio-app/gradio#12992)
    # footer_links=[] removes the footer from the DOM, causing infinite vertical growth
    # Hiding via CSS keeps the element in the DOM as an anchor for the iframe height calculation.
    "footer { height: 5px !important; visibility: hidden !important; }\n"

    # Preload dark-mode backgrounds to prevent white flash before Gradio's CSS loads
    # Values from "origin" theme: body=#0b0f19 (neutral_950), .bubble-wrap=#111827 (neutral_900)
    "@media (prefers-color-scheme: dark) {\n"
    "  .gradio-container { background: #0b0f19 !important; }\n"  # page background
    "  .bubble-wrap { background: #111827 !important; }\n"       # chatbox
    "  .bot.message { background: #111827 !important; }\n"       # bot messages (viz., greeting)
    "  div.form { background: #1f2937 !important; }\n"           # input box XXX not working
    "  div.row { background: #1f2937 !important; }\n"            # input box XXX not working
    "  .gr-group { background: #1f2937 !important; }\n"          # input box XXX not working
    "  .input-container { background: #1f2937 !important; }\n"   # input box XXX not working
    "  textarea { background: #1f2937 !important; }\n"           # input box XXX not working
    "}\n"
)

class APIRequestLogger(BaseHTTPMiddleware):
    """Log requests to Gradio API endpoints, flagging likely non-browser access."""
    _logger = logging.getLogger("request")

    async def dispatch(self, request, call_next):
        # only log interesting endpoints (even static asset requests come through here)
        if request.url.path.startswith(('/gradio_api/call/', '/gradio_api/queue/join')):
            src = "browser" if request.headers.get("sec-fetch-mode") is not None else "DIRECT"
            host = request.client.host if request.client is not None else "UNKNOWN"
            self._logger.info("%s %s [%s] from %s", request.method, request.url.path, src, host)
        return await call_next(request)

if __name__ == "__main__":
    demo.launch(
        footer_links=['settings'],  # if footer_links is empty, the iframe grows without bound
                                    # bug reported at gradio-app/gradio#12992 (Gradio 6.9.0)
                                    # we keep this minimal, then hide the footer with custom_css
        favicon_path=config.BASE_DIR / 'assets' / 'favicon.ico',
        theme="origin",
        css=custom_css,
        app_kwargs={"middleware": [Middleware(APIRequestLogger)]},
    )
