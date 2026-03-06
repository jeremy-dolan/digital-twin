import os
from random import randint
from typing import Callable, TypedDict

import requests
from openai.types.responses import FunctionToolParam

import config


class ToolEntry(TypedDict):
    spec: FunctionToolParam
    fn: Callable[..., str]


class ToolRegistry:
    """
    Class to register tools available to the LLM.
    Each tool requires a `FunctionToolParam` spec to provide to the LLM,
    and a function to run when invoked by the model.
    """
    def __init__(self):
        self.tools: dict[str, ToolEntry] = {}
    def add(self, spec: FunctionToolParam, fn: Callable[..., str]):
        """Register a tool (under the name given in `spec`)."""
        self.tools[spec["name"]] = {"spec": spec, "fn": fn}
    def get_specs(self, tools: list[str] | None = None) -> list[FunctionToolParam]:
        """Return function specs for the given tool names (or all tools, if none specified)."""
        if not tools:
            tools = list(self.tools.keys())
        return [self.tools[name]["spec"] for name in tools if name in self.tools]


### tool processing functions

def llm_send_notification(message: str) -> str:
    """Send a push notification via the Pushover service."""
    payload = {
        "user": os.getenv("PUSHOVER_USER"),
        "token": os.getenv("PUSHOVER_TOKEN"),
        "message": message,
    }
    requests.post(config.PUSHOVER_ENDPOINT, data=payload)
    # FIXME - check return code and any other status returned in JSON package
    return "Message sent."


def llm_roll_dice() -> str:
    """Get a 1d6 random value. To demonstrate sequential tool calls."""
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
    """Create and return a ToolRegistry with the full list of tools."""
    tools = ToolRegistry()
    tools.add(SEND_NOTIFICATION_SPEC, llm_send_notification)
    tools.add(ROLL_DICE_SPEC, llm_roll_dice)
    return tools
