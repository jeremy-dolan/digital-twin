import json
import threading

from openai import OpenAI, APIError
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
)

import config
import prompts
from tools import ToolEntry, ToolRegistry, llm_send_notification


def _process_tool_calls(
    response_output_items: list[ResponseOutputItem],
    tool_registry: ToolRegistry,
    messages: list[ResponseInputItemParam],
) -> int:
    """Process any tool calls in a response; append results to message list in place.
    Returns the number of tool calls processed."""
    tool_calls = 0
    for item in response_output_items:
        if item.type != "function_call":
            continue
        if item.name not in tool_registry:
            print(f'WARNING: model tried calling unknown tool {item.name} with args: {item.arguments}')
            continue
        tool_calls += 1
        tool_result = tool_registry[item.name]['fn'](**json.loads(item.arguments))
        messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(tool_result),
        })
    return tool_calls


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
        print('WARNING: cannot send summary notification (send_notification not registered)')
        return

    summary_corpus = _normalize_mixed_history(messages)[-20:]  # 20 most recent user/assistant msgs
    # summary_corpus.insert(0, {"role": "developer", "content": prompts.BG_SUMMARY})

    try:
        resp = client.responses.create(
            model=config.INFERENCE_MODEL,
            instructions=prompts.SUMMARY_NOTIFICATION,
            input=summary_corpus,
        )
        tool_registry['send_notification']['fn'](message=resp.output_text)
    except Exception as e: # XXX/TODO: compare other responses try/except
        print(f"Background summary notification failed: {e}")


def resolve_turn(
    client: OpenAI,
    input_messages: list[ResponseInputItemParam],
    tool_registry: ToolRegistry,
) -> str:
    """Process a conversation turn: send messages to the LLM, handle all tool calls (up to `config.MAX_CONSECUTIVE_TOOL_CALLS`), and return the model's final text response."""
    messages = list(input_messages)    # shallow copy to avoid side effects
    tools = tool_registry.get_specs()  # all registered tools
    loop_count = 0

    while True:
        loop_count += 1
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS:
            tools = []
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS + 1:
            print(f"WARNING: exceeded {config.MAX_SEQUENTIAL_TOOL_CALLS} tool calls")
            break

        try:
            resp = client.responses.create(
                model=config.INFERENCE_MODEL,
                input=messages,
                tools=tools,
                reasoning=config.REASONING,
                text={'verbosity': 'low'},
            )
        except APIError as e:
            print(f"OpenAI call failed: {type(e).__name__}: {e}")
            return "Oof, sorry, technical hiccup on my end. Try asking again in a sec."

        # may include: ResponseOutputMessage, ResponseReasoningItem, ResponseFunctionToolCall...
        # TODO: we should stream, then display ResponseReasoningItem's as:
        #              role="assistant",
        #              metadata={"title": "⏳Thinking: **ResponseReasoningItem short summary**}
        messages.extend(resp.output) # type: ignore
                                     # (ResponseOutputItem guaranteed valid ResponseInputItemParam)

        if _process_tool_calls(resp.output, tool_registry, messages) == 0:
            break
        
    # every second message, update me with a conversation summary notification (sent in background)
    user_m_count = len([m for m in messages if isinstance(m, dict) and m.get('role') == 'user'])
    if user_m_count % 2 == 0:
        threading.Thread(
            target=_summary_notification_daemon,
            args=(client, messages, tool_registry),
            daemon=True,
        ).start()

    # return the text of the final Response (tool I/O and reasoning summaires are abandoned)
    return resp.output_text
