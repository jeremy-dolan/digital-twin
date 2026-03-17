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
    Manages a 'Thinking...' accordion in the Gradio chat UI to display reasoning summaries and tool
    calls. The accordion is a gr.ChatMessage with a `metadata` attribute, which causes it to render
    as a separate bubble attached to the subsequent assistant message. We asynchronously accumulate
    reasoning summaries and tool calls/results as the content of this message.

    All methods mutate `ui_messages` in place. Caller should yield it after each update.
    """

    def __init__(self, ui_messages: list[ChatMessage]):
        self._messages = ui_messages        # XXX a vestige of nested thoughts. Arg still needed?
        self._msg_index: int | None = None
        self._parts: dict[str, str] = {}  # ordered dict of 'thoughts'
        self._start = time.time()
        self._meta: MetadataDict = {"title": "🤔 Thinking...", "status": "pending"}  # 🤔🧐🤨⏳💡💭
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
        """Keep accordion open; replace ellipsis and spinner with duration."""
        if self._msg_index is not None and not self.finalized:
            self.finalized = True
            del self._meta["status"]  # omit status to keep accordion open (without a spinner)
            self._meta["title"] = "🤔 Thinking"  # remove the ellipsis
            self._meta["duration"] = round(time.time() - self._start, 2)
            self._render()

    def _render(self):
        r"""Turn all `_parts` into \n-separated entries of thought content."""
        # XXX I think this code is just bad;
        # because we're operating on ui_messages (which is new each stream)
        # the accordian (if it exists) will always be index 0
        assert self._msg_index is None or self._msg_index == 0
        # use with the above for now to double check, until I get to refactoring this
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
        logger.warning('cannot send summary notification: send_notification not registered')
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
    api_messages: list[ResponseInputItemParam],
    tool_registry: ToolRegistry,
) -> Generator[tuple[list[ChatMessage], list[ResponseInputItemParam]], None, None]:
    """
    Stream a conversation turn, yielding (new_ui_msgs, api_messages) tuples.
    Handles tool calls by executing them and re-streaming for the model's next response.
    Reasoning summaries and tool usage are shown in a single collapsible `_ThoughtAccordion`
    """
    api_messages = list(api_messages)    # shallow copy; caller gets final state via yield
    new_ui_msgs: list[ChatMessage] = []  # accumulated Gradio messages for this turn, to update UI
    tools = tool_registry.get_specs()    # for the API; specs for all registered tools
    loop_count = 0
    thinking = _ThoughtAccordion(new_ui_msgs)

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
                input=api_messages,
                tools=tools,
                reasoning={'effort': 'medium', 'summary': 'concise'},
                include=["reasoning.encrypted_content"],
                text={'verbosity': 'low'},  # helps keep model on-topic
                stream=True,
            )
        except APIError as e:
            logger.error("OpenAI call failed: %s: %s", type(e).__name__, e)
            new_ui_msgs.append(ChatMessage(role="assistant", content=IN_CHARACTER_ERROR))
            yield new_ui_msgs, api_messages
            break

        # per-stream (per model call) state (resets each time we get a new response after tool use)
        response_text = ""
        response_text_initiated = False
        has_tool_calls = False

        try:
            for event in stream:
                # we only catch the response events we care about, ignoring many, including:
                #  .created, .in_progress, .function_call_arguments.delta, .output_item.added,
                #  .content_part.added, .output_text.done, .content_part.done, .output_item.done
                if event.type == 'response.reasoning_summary_text.delta':
                    key = f'r_{loop_count}_{event.output_index}_{event.summary_index}'
                    thinking.add_reasoning_delta(key, event.delta)
                    yield new_ui_msgs, api_messages

                elif event.type == 'response.function_call_arguments.done':
                    thinking.set_tool_pending(event.item_id, event.name)
                    yield new_ui_msgs, api_messages

                elif event.type == 'response.output_text.delta':
                    if not response_text_initiated:
                        response_text_initiated = True
                        # first output_text; model is done with reasoning/tool calling this stream
                        thinking.finalize()
                        new_ui_msgs.append(ChatMessage(role="assistant", content=""))

                    response_text += event.delta
                    new_ui_msgs[-1].content = response_text
                    yield new_ui_msgs, api_messages

                elif event.type == 'response.completed':
                    # This stream is done. If there are tool calls, we process them and re-stream

                    response = event.response
                    # accumulate this stream's responses onto the API message history
                    # (ResponseReasoningItem, ResponseFunctionToolCall, ResponseOutputMessage)
                    api_messages.extend(response.output)  # type: ignore (ResponseOutputItems are
                                                          # valid ResponseInputItemParams)

                    # execute tool calls from this stream, update thought accordion with results
                    for item in response.output:
                        if item.type != "function_call":
                            continue
                        if item.name not in tool_registry:
                            logger.warning('model tried calling unknown tool %s with args: %s',
                                           item.name, item.arguments)
                            continue
                        has_tool_calls = True
                        tool_result = tool_registry[item.name]['fn'](**json.loads(item.arguments))
                        api_messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(tool_result),
                        })
                        thinking.set_tool_result(item.id, item.name, tool_result)  # type: ignore
                        yield new_ui_msgs, api_messages

        except APIError as e:
            logger.error("OpenAI stream error: %s: %s", type(e).__name__, e)
            new_ui_msgs.append(ChatMessage(role="assistant", content=IN_CHARACTER_ERROR))
            yield new_ui_msgs, api_messages
            break

        if not has_tool_calls:
            break
        # else: tool calls were answered, so we loop to stream another response from the model


    # cleanup after `break`
    if not thinking.finalized:
        thinking.finalize()
        yield new_ui_msgs, api_messages

    # every other user message, update me with a conversation summary (run in background)
    user_m_count = len([m for m in api_messages if isinstance(m, dict) and m.get('role') == 'user'])
    if user_m_count % 2 == 0:
        threading.Thread(
            target=_summary_notification_daemon,
            args=(client, api_messages, tool_registry),
            daemon=True,
        ).start()
