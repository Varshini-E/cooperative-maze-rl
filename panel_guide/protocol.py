"""Small, strict wire protocol. Model prose never directly executes an action."""

import json
import re
from dataclasses import dataclass


class ProtocolError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_output") -> None:
        super().__init__(message)
        self.code = code


OPERATOR_RESPONSES = {
    "ask": ("reply", "unclear"),
    # An instruct turn may still be answered with a reply when the Expert mislabels a question.
    "instruct": ("move", "toggle", "satisfied", "unclear", "reply"),
}
TARGET_PATTERNS = {"move": r"^door_[1-9][0-9]*$", "toggle": r"^control_[1-9][0-9]*$"}
NOTES_LIMIT = 300  # Expert's private note: short, like a human jotting on paper.


def operator_response_schema(turn: str) -> dict:
    """Advertised turn contract; no room state or target availability is exposed."""
    if turn not in OPERATOR_RESPONSES:
        raise ValueError("Operator turn must be ask or instruct.")
    options = []
    for kind in OPERATOR_RESPONSES[turn]:
        action = kind in TARGET_PATTERNS
        key = "target" if action else "text"
        value = {"type": "string", "minLength": 1, "maxLength": 64 if action else 2048,
                 "pattern": TARGET_PATTERNS[kind] if action else r"\S"}
        options.append({"type": "object", "properties": {
            "type": {"const": kind}, key: value,
        }, "required": ["type", key], "additionalProperties": False})
    return {"oneOf": options}


@dataclass(frozen=True)
class Message:
    type: str
    value: str
    # Expert-private note. Never part of the shared transcript or to_dict().
    notes: str | None = None

    def to_dict(self) -> dict[str, str]:
        key = "target" if self.type in ("move", "toggle") else "text"
        return {"type": self.type, key: self.value}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON key.")
        result[key] = value
    return result


def parse_message(raw: str, role: str, *, operator_turn: str | None = None,
                  allow_notes: bool = False) -> Message:
    if role not in ("expert", "operator"):
        raise ValueError("Unknown role.")
    if operator_turn is not None and (role != "operator" or operator_turn not in OPERATOR_RESPONSES):
        raise ValueError("A turn contract applies only to an Operator answering ask or instruct.")
    limit = 4096 if role == "expert" else 2048
    if not isinstance(raw, str) or len(raw) > limit:
        raise ProtocolError("Response exceeds the character limit.")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("Return one JSON object with no Markdown or duplicate keys.") from exc
    allowed = ("ask", "instruct") if role == "expert" else OPERATOR_RESPONSES["instruct"]
    if not isinstance(data, dict) or data.get("type") not in allowed:
        raise ProtocolError("Unknown message type.")
    kind = data["type"]
    key = "target" if kind in ("move", "toggle") else "text"
    notes = None
    if allow_notes and role == "expert" and "notes" in data:
        notes = data.pop("notes")
        if not isinstance(notes, str) or not notes.strip() or len(notes) > NOTES_LIMIT:
            raise ProtocolError(f"notes must be a nonempty string of at most {NOTES_LIMIT} characters.")
    if set(data) != {"type", key} or not isinstance(data[key], str) or not data[key].strip():
        raise ProtocolError(f"Expected only type and a nonempty {key} string.")
    if key == "target" and len(data[key]) > 64:
        raise ProtocolError("Target ID is too long.")
    if operator_turn is not None:
        responses = OPERATOR_RESPONSES[operator_turn]
        if kind not in responses:
            code = "instruction_following_failure" if operator_turn == "instruct" else "invalid_output"
            raise ProtocolError(
                f"An {operator_turn} turn permits only {', '.join(responses)}.",
                code=code,
            )
        if kind in TARGET_PATTERNS and re.fullmatch(TARGET_PATTERNS[kind], data[key]) is None:
            raise ProtocolError(f"Invalid {kind} argument: target must match {TARGET_PATTERNS[kind]}.")
    return Message(kind, data[key], notes)
