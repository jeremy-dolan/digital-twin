import json
import logging
import time
import threading
from collections.abc import Generator

from gradio import ChatMessage
from gradio.components.chatbot import MetadataDict
from openai import OpenAI, APIError
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputText,
)

import config
import prompts
from tools import ToolRegistry

logger = logging.getLogger(__name__)

IN_CHARACTER_ERROR = "Oof, sorry, technical hiccup on my end. Try asking again in a sec."


class _ThoughtAccordion:
    """
    Manages a unified 'Thinking...' accordion in the Gradio chat UI.

    Reasoning summaries and tool calls accumulate as content inside a single element. The accordion
    is open with a spinner while pending, and stays open (no spinner, with duration) once finalized.

    Mutates `chat_messages` in place. Caller should yield it after each update.
    """

    def __init__(self, chat_messages: list[ChatMessage]):
        self._messages = chat_messages
        self._msg_index: int | None = None
        self._parts: dict[str, str] = {}  # ordered dict of content lines
        self._start = time.time()
        self._meta: MetadataDict = {"title": "🤔 Thinking...", "status": "pending"} # 🤔🧐🤨⏳💡💭
        self.finalized = False

    def add_reasoning_delta(self, key: str, delta: str):
        """Accumulate streaming reasoning summary text under `key`."""
        self._parts[key] = self._parts.get(key, "") + delta
        self._render()

    def set_tool_pending(self, item_id: str, name: str | None):
        """Show a tool call as in-progress."""
        self._parts[f"t_{item_id}"] = f"🔧 {name or 'tool'}..."
        self._render()

    def set_tool_result(self, item_id: str, name: str, result: str):
        """Update a tool call line with its result."""
        self._parts[f"t_{item_id}"] = f"🔧 {name}: {result}"
        self._render()

    def finalize(self):
        """Close the spinner, show duration. Accordion stays expanded (status omitted)."""
        if self._msg_index is not None and not self.finalized:
            self.finalized = True
            self._meta["title"] = "🤔 Thinking" # remove the ellipsis
            del self._meta["status"] # omit status to keep accordion open (without spinner)
            self._meta["duration"] = round(time.time() - self._start, 2)
            self._render()

    def _render(self):
        content = "\n".join(self._parts.values())
        msg = ChatMessage(role="assistant", content=content, metadata=self._meta)
        if self._msg_index is None:
            self._msg_index = len(self._messages)
            self._messages.append(msg)
        else:
            self._messages[self._msg_index] = msg


def _normalize_mixed_history(messages):
    """Build a normalized dict of ONLY user and assistant message texts. Drops context
    injections (role=developer), tool calls, reasoning summaries, and message metadata."""
    normed = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m:
            normed.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m, ResponseOutputMessage) and isinstance(m.content[0], ResponseOutputText):
            normed.append({"role": m.role, "content": m.content[0].text})
    return normed

def _summary_notification_daemon(
    client: OpenAI,
    messages: list[ResponseInputItemParam],
    tool_registry: ToolRegistry,
) -> None:
    """
    Summarize the conversation so-far and send a push notification using the 'send_notification'
    tool. Intended to run as a daemon thread so it doesn't block user-facing response.
    """
    if 'send_notification' not in tool_registry:
        logger.warning('cannot send summary notification (send_notification not registered)')
        return

    summary_corpus = _normalize_mixed_history(messages)[-20:]  # 20 most recent user/assistant msgs

    try:
        resp = client.responses.create(
            model=config.INFERENCE_MODEL,
            instructions=prompts.SUMMARY_NOTIFICATION,
            input=summary_corpus,
        )
        tool_registry['send_notification']['fn'](message=resp.output_text)
    except Exception as e:
        logger.error("Background summary notification failed", exc_info=True)


def stream_turn(
    client: OpenAI,
    input_messages: list[ResponseInputItemParam],
    tool_registry: ToolRegistry,
) -> Generator[list[ChatMessage], None, None]:
    """
    Stream a conversation turn, yielding lists of ChatMessage for Gradio UI updates.
    Handles tool calls by executing them and re-streaming for the model's next response.
    Reasoning summaries and tool usage are shown in a single collapsible `_ThoughtAccordion`
    """
    messages = list(input_messages)        # shallow copy to avoid side effects
    tools = tool_registry.get_specs()      # all registered tools
    ui_messages: list[ChatMessage] = [] # accumulated UI messages for this turn
    loop_count = 0
    thinking = _ThoughtAccordion(ui_messages)

    while True:
        loop_count += 1
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS:
            tools = []
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS + 1:
            logger.warning("exceeded %s sequential tool calls", config.MAX_SEQUENTIAL_TOOL_CALLS)
            break

        try:
            stream = client.responses.create(
                model=config.INFERENCE_MODEL,
                input=messages,
                tools=tools,
                reasoning=config.REASONING,
                text={'verbosity': 'low'},  # helps keep model on-topic
                stream=True,
            )
        except APIError as e:
            logger.error("OpenAI call failed: %s: %s", type(e).__name__, e)
            ui_messages.append(ChatMessage(role="assistant", content=IN_CHARACTER_ERROR))
            yield ui_messages
            return

        # Per-stream-iteration state (reset each time we re-call the API after tool use)
        response_text = ""
        response_msg_idx: int | None = None
        has_tool_calls = False

        try:
            for event in stream:
                if event.type == 'response.reasoning_summary_text.delta':
                    key = f"r_{loop_count}_{event.output_index}_{event.summary_index}"
                    thinking.add_reasoning_delta(key, event.delta)
                    yield ui_messages

                elif event.type == 'response.function_call_arguments.done':
                    thinking.set_tool_pending(event.item_id, event.name)
                    yield ui_messages

                elif event.type == 'response.output_text.delta':
                    if not thinking.finalized:
                        thinking.finalize()
                    response_text += event.delta
                    if response_msg_idx is None:
                        response_msg_idx = len(ui_messages)
                        ui_messages.append(ChatMessage(
                            role="assistant", content=response_text,
                        ))
                    else:
                        ui_messages[response_msg_idx] = ChatMessage(
                            role="assistant", content=response_text,
                        )
                    yield ui_messages

                elif event.type == 'response.completed':
                    response = event.response
                    messages.extend(response.output)  # type: ignore
                                                      # (ResponseOutputItem guaranteed valid
                                                      #  ResponseInputItemParam)

                    # Execute tool calls and update their lines in the thought accordion
                    for item in response.output:
                        if item.type != "function_call":
                            continue
                        if item.name not in tool_registry:
                            logger.warning('model tried calling unknown tool %s with args: %s',
                                           item.name, item.arguments)
                            continue
                        has_tool_calls = True
                        tool_result = tool_registry[item.name]['fn'](**json.loads(item.arguments))
                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(tool_result),
                        })
                        thinking.set_tool_result(item.id, item.name, tool_result)
                        yield ui_messages

        except APIError as e:
            logger.error("OpenAI stream error: %s: %s", type(e).__name__, e)
            ui_messages.append(ChatMessage(role="assistant", content=IN_CHARACTER_ERROR))
            yield ui_messages
            return

        if not has_tool_calls:
            break
        # Tool calls were processed; loop to stream the model's next response

    # Finalize thought if it wasn't already (e.g. tool-only turn with no text response)
    if not thinking.finalized:
        thinking.finalize()
        yield ui_messages

    # Every second message, update me with a conversation summary notification (in background)
    user_m_count = len([m for m in messages if isinstance(m, dict) and m.get('role') == 'user'])
    if user_m_count % 2 == 0:
        threading.Thread(
            target=_summary_notification_daemon,
            args=(client, messages, tool_registry),
            daemon=True,
        ).start()
