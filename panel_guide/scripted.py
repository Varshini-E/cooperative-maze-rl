"""Free plumbing check for the hand-written fixture; not an experimental baseline."""

import json
import re
from dataclasses import asdict

from panel_guide.conversation import Completion, Request
from panel_guide.curriculum import REFERENCE_ROUTE
from panel_guide.protocol import Message


class ScriptedExpert:
    # Semantic route for the fixture. No physical slot IDs or solver calls.
    route = REFERENCE_ROUTE

    def __init__(self, route: tuple[tuple[str, str], ...] | None = None) -> None:
        self.route = self.route if route is None else route
        self.turn = 0

    async def complete(self, request: Request) -> Completion:
        # If the partner did not complete the fixed route, finish under the game
        # budget instead of misclassifying a script exhaustion as an API outage.
        if self.turn % 2 == 0 or self.turn // 2 >= len(self.route):
            message = Message("ask", "Describe the current room's emblem, every local door, and every local control.")
        else:
            index = self.turn // 2
            kind, symbol = self.route[index]
            if kind == "move":
                text = f"Move through the door with symbol: {symbol}"
            else:
                text = f"Ensure this control is ON; toggle only if OFF. Symbol: {symbol}"
            message = Message("instruct", text)
        self.turn += 1
        return Completion(message.to_json())


COMMAND_CONTRACT = """
For this diagnostic, your partner is a deterministic executor with perfect local
observations. An ask returns every visible room field. For instruct, text must be
exactly 'move door_N' or 'toggle control_N', using a currently observed target ID.
Other instruction formats receive a clarification. It has no map, hidden switch
values, or route planner. You still decide where to go and which switch to change.
This scripted diagnostic reports only current-room fields, even when the visual
renderer shows a whole floor plan. It is not a full-map perception baseline.
"""


class DiagnosticOperator:
    """Strict command grammar avoids pretending a script understands arbitrary prose."""

    async def complete(self, request: Request) -> Completion:
        view = request.scripted_view
        if view is None:
            raise ValueError("A visible-room observation is required.")
        current = next(event for event in reversed(request.history) if event["role"] == "expert")
        if current["type"] == "ask":
            message = Message("reply", json.dumps(asdict(view)))
        else:
            match = re.fullmatch(r"(move) (door_\d+)|(toggle) (control_\d+)", current["text"].strip())
            if match is None:
                message = Message("unclear", "No action executed. Use exactly: move door_N or toggle control_N.")
            else:
                kind = match[1] or match[3]
                target = match[2] or match[4]
                visible = view.doors if kind == "move" else view.controls
                item = next((item for item in visible if item.id == target), None)
                if item is None:
                    message = Message("unclear", "No action executed: that target is not visible in this room.")
                elif kind == "move" and not item.is_open:
                    message = Message("unclear", "No action executed: that door is locked.")
                else:
                    message = Message(kind, target)
        return Completion(message.to_json())


class ScriptedOperator:
    async def complete(self, request: Request) -> Completion:
        view = request.scripted_view
        if view is None:
            raise ValueError("The scripted Operator needs the permitted visible-room fields.")
        expert = next(event for event in reversed(request.history) if event["role"] == "expert")
        if expert["type"] == "ask":
            message = Message("reply", json.dumps(asdict(view)))
        elif expert["text"].startswith("Move through the door with symbol: "):
            symbol = expert["text"].split(": ", 1)[1]
            door = next((d for d in view.doors if d.symbol == symbol), None)
            message = Message("move", door.id) if door else Message("unclear", "That door is not visible.")
        elif expert["text"].startswith("Ensure this control is ON; toggle only if OFF. Symbol: "):
            symbol = expert["text"].split("Symbol: ", 1)[1]
            control = next((c for c in view.controls if c.symbol == symbol), None)
            if control is None:
                message = Message("unclear", "That control is not visible.")
            elif control.is_on:
                message = Message("satisfied" if request.protocol_version in ("v4", "v6") else "unclear",
                                  "The control is already ON; no toggle is needed.")
            else:
                message = Message("toggle", control.id)
        else:
            message = Message("unclear", "The fixture script does not recognize this instruction.")
        return Completion(message.to_json())
