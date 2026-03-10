import os
from random import randint
from typing import Callable, TypedDict, Iterator
from dataclasses import dataclass, field

import requests
from openai.types.responses import FunctionToolParam

import config


class ToolEntry(TypedDict):
    spec: FunctionToolParam
    fn: Callable[..., str]

@dataclass
class ToolRegistry:
    """
    Class to register tools to make available to the LLM. Each tool requires a `FunctionToolParam`
    spec to provide to the LLM, and a `Callable` to run when invoked by the model.
    """
    _tools: dict[str, ToolEntry] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
    def __getitem__(self, name: str) -> ToolEntry:
        return self._tools[name]
    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)
    def add(self, spec: FunctionToolParam, fn: Callable[..., str]):
        """Register a tool (under the name given in `spec`)."""
        self._tools[spec["name"]] = {"spec": spec, "fn": fn}
    def subset(self, names: list[str]) -> "ToolRegistry":
        """Return a new `ToolRegistry` narrowed to just the specified tool names."""
        return ToolRegistry( {name: self._tools[name] for name in names if name in self} )
    def get_specs(self, names: list[str] | None = None) -> list[FunctionToolParam]:
        """Return function specs for tool `names` (or for all tools, if none are specified)."""
        return [ self[name]["spec"] for name in (names or self) if name in self ]


### tool processing functions

def llm_send_notification(message: str) -> str:
    """
    Send a push notification via the Pushover service.
    API documentation: https://pushover.net/api
    """
    payload = {
        "user": os.getenv("PUSHOVER_USER"),
        "token": os.getenv("PUSHOVER_TOKEN"),
        "message": message,
    }
    response = requests.post(config.PUSHOVER_ENDPOINT, data=payload, timeout=8)

    if response.ok and response.json().get('status') == 1:
        return "Push notification sent successfully."
    else:
        print(f'send_notification failed with HTTP status {response.status_code}: {response.text}')
        return "Error. Unable to send notification at this time."


def llm_roll_dice() -> str:
    """Get a 1d6 random value. Useful for testing sequential tool calls."""
    return str(randint(1, 6))


### tool specifications

SEND_NOTIFICATION_SPEC: FunctionToolParam = {
    "type": "function",
    "strict": True,
    "name": "send_notification",
    "description": "Sends a push notification to the real-world version of you via the "
                   "Pushover service. Use this if the user needs to alert the "
                   "real-world version of you.",
                   # or to notify him about about important events, completed tasks,
                   # or time-sensitive information.
    "parameters": { # JSON Schema
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The notification message to send to the user's device",
            },
        },
        "required": ["message"],
        "additionalProperties": False,
    },
}

ROLL_DICE_SPEC: FunctionToolParam = {
    "type": "function",
    "strict": True,
    "name": "roll_dice",
    "description": "Roll a single six-sided dice and return the value.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}


def build_all_tools() -> ToolRegistry:
    """Create and return a ToolRegistry with all available tools."""
    tools = ToolRegistry()

    if (os.getenv("PUSHOVER_USER") is None or os.getenv("PUSHOVER_TOKEN") is None):
        print("WARNING: Pushover credentials not available, cannot register send_notification tool")
    else:
        tools.add(SEND_NOTIFICATION_SPEC, llm_send_notification)

    tools.add(ROLL_DICE_SPEC, llm_roll_dice)
    return tools
