import json

from openai import OpenAI
from openai.types.responses import ResponseOutputItem, ResponseInputItemParam

import config
from tools import ToolRegistry


def _process_tool_calls(
    response_items: list[ResponseOutputItem],
    tool_registry: ToolRegistry,
    messages: list[ResponseInputItemParam],
) -> int:
    """Process any tool calls in a response; append results to message list in place.
    Returns the number of tool calls processed."""
    tool_calls = 0
    for item in response_items:
        if item.type != "function_call":
            continue
        if item.name not in tool_registry.tools.keys():
            print(f'WARNING: model tried calling unknown tool {item.name} with args: {item.arguments}')
            continue
        tool_calls += 1
        tool_result = tool_registry.tools[item.name]['fn'](**json.loads(item.arguments))
        messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(tool_result),
        })
    return tool_calls


def resolve_turn(
    client: OpenAI,
    input_messages: list[ResponseInputItemParam],
    tool_registry: ToolRegistry,
) -> str:
    """Process a conversation turn: send messages to the LLM, handle all tool calls (up to `config.MAX_CONSECUTIVE_TOOL_CALLS`), and return the model's final text response."""
    messages = list(input_messages)  # shallow copy to avoid side effects
    tools = tool_registry.get_specs()
    loop_count = 0

    while True:
        loop_count += 1
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS:
            tools = []
        if loop_count > config.MAX_SEQUENTIAL_TOOL_CALLS + 1:
            print(f"WARNING: exceeded {config.MAX_SEQUENTIAL_TOOL_CALLS} tool calls")
            break

        resp = client.responses.create(
            model=config.INFERENCE_MODEL,
            input=messages,
            tools=tools,
            reasoning=config.REASONING,
        )
        
        messages.extend(resp.output) # type: ignore (ResponseOutputItem is a valid ResponseInputItemParam)

        if _process_tool_calls(resp.output, tool_registry, messages) == 0:
            break

    return resp.output_text
