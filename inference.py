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
    Manages a ChatMessage that renders as a 'Thinking...' accordion attached to an assistant
    message. Accumulates reasoning summary deltas and tool calls/results. Caller should append
    `.chatmessage` to the UI message list; subsequent methods mutate that ChatMessage in place.
    """
    def __init__(self):
        self._parts: dict[str, str] = {}  # ordered dict of 'thoughts'
        self._start = time.time()
        self._meta: MetadataDict = {"title": "💭 Thinking...", "status": "pending"}  # 🤔🧐🤨⏳💡💭
        self.chatmessage = ChatMessage(role="assistant", content="", metadata=self._meta)
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
        """Keep accordion open; remove ellipsis and spinner; add duration."""
        if not self.finalized:
            self.finalized = True
            del self._meta["status"]  # omit status to keep accordion open (without a spinner)
            self._meta["title"] = "💭 Thinking"  # remove the ellipsis
            self._meta["duration"] = round(time.time() - self._start, 2)

    def _render(self):
        """Update content from accumulated parts."""
        self.chatmessage.content = "\n".join(self._parts.values())


def _normalize_mixed_history(messages):
    """Build a normalized dict of ONLY user and assistant message texts. Drops context
    injections (role=developer), function calls, reasoning summaries, and message metadata."""
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
    tool_registry: ToolRegistry,
    new_ui_msgs: list[ChatMessage],
    api_messages: list[ResponseInputItemParam],
) -> Generator[tuple[list[ChatMessage], list[ResponseInputItemParam]], None, None]:  # yield only
    """
    Stream the next conversation turn from the model based on conversation state in `api_messages`.
    Handles tool calls (execution and LLM follow-up) for tools in the provided ToolRegistry.
    Reasoning summaries and tool usage are shown in a collapsible `_ThoughtAccordion`.
    Caller may seed `new_ui_msgs` with other messages for this callback (e.g. a RAG accordion).
    Yields a tuple back to Gradio's session management: this turn's ChatMessage additions for
    the UI (`new_ui_msgs`), and the entire session's `api_messages`.
    """
    _debug_log_api_messages(api_messages)

    tools = tool_registry.get_specs()    # for the API; specs for all registered tools
    loop_count = 0

    thinking = _ThoughtAccordion()
    thinking_visible = False

    while True:
        loop_count += 1
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS:
            # stop advertising tools
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
                # we only catch a less-than-perfectly-robust subset of events
                # see dev/response-events.md for more details on event types
                if event.type == 'response.reasoning_summary_text.delta':
                    if not thinking_visible:
                        thinking_visible = True
                        new_ui_msgs.append(thinking.chatmessage)
                    key = f'r_{loop_count}_{event.output_index}_{event.summary_index}'
                    thinking.add_reasoning_delta(key, event.delta)
                    yield new_ui_msgs, api_messages

                elif event.type == 'response.function_call_arguments.done':
                    if not thinking_visible:
                        thinking_visible = True
                        new_ui_msgs.append(thinking.chatmessage)
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
                    # add all of this stream's response objects to the API message history
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

        except GeneratorExit:
            # If a user presses cancel mid-stream, Gradio should .close the generator so we can
            # catch here and close the stream. In Gradio 6.9.0, cancel causes a RuntimeWarning.
            # Should have been fixed, but wasn't, by gradio-app/gradio#11396.
            # I filed gradio-app/gradio#13044 which was fixed for Gradio 6.10.0 (TEST THIS)
            stream.close()
            logger.info("Client cancelled stream — closed OpenAI connection")
            return
        except APIError as e:
            logger.error("OpenAI stream error: %s: %s", type(e).__name__, e)
            new_ui_msgs.append(ChatMessage(role="assistant", content=IN_CHARACTER_ERROR))
            yield new_ui_msgs, api_messages
            break
        finally:
            stream.close()

        if not has_tool_calls:
            break
        # else: tool calls were answered, so we loop and stream another response from the model


    # cleanup after `break`
    if thinking_visible and not thinking.finalized:
        thinking.finalize()
        yield new_ui_msgs, api_messages

    # every other user message, update me with a conversation summary (run in background)
    # todo: thread could check if "interesting," and call tool
    user_m_count = len([m for m in api_messages if isinstance(m, dict) and m.get('role') == 'user'])
    if user_m_count % 2 == 0:
        threading.Thread(
            target=_summary_notification_daemon,
            args=(client, api_messages, tool_registry),
            daemon=True,
        ).start()


def _debug_log_api_messages(msgs):
    logger.debug('--- stream_turn received API messages: ---')
    # truncate presumptive message prompt:
    logger.debug('%s', {**msgs[0], 'content': msgs[0]['content'][:40] + '...'})
    for m in msgs[1:]:
        logger.debug('%s', m)
    logger.debug('------------------------------------------')
