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
    "instruct": ("move", "toggle", "unclear"),
}
TARGET_PATTERNS = {"move": r"^door_[1-9][0-9]*$", "toggle": r"^control_[1-9][0-9]*$"}


def operator_responses(turn: str, protocol_version: str = "v3") -> tuple[str, ...]:
    if protocol_version not in ("v3", "v4", "v6"):
        raise ValueError("protocol_version must be v3, v4, or v6 (prompt v5 uses v4).")
    if turn not in OPERATOR_RESPONSES:
        raise ValueError("Operator turn must be ask or instruct.")
    if protocol_version == "v6" and turn == "instruct":
        return ("move", "toggle", "satisfied", "unclear", "reply")
    if protocol_version == "v4" and turn == "instruct":
        return ("move", "toggle", "satisfied", "unclear")
    return OPERATOR_RESPONSES[turn]


def operator_response_schema(turn: str, protocol_version: str = "v3") -> dict:
    """Advertised turn contract; no room state or target availability is exposed."""
    options = []
    for kind in operator_responses(turn, protocol_version):
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
                  protocol_version: str = "v3") -> Message:
    if role not in ("expert", "operator"):
        raise ValueError("Unknown role.")
    if operator_turn is not None and (role != "operator" or operator_turn not in OPERATOR_RESPONSES):
        raise ValueError("A turn contract applies only to an Operator answering ask or instruct.")
    if protocol_version not in ("v3", "v4", "v6"):
        raise ValueError("protocol_version must be v3, v4, or v6 (prompt v5 uses v4).")
    limit = 4096 if role == "expert" else 2048
    if not isinstance(raw, str) or len(raw) > limit:
        raise ProtocolError("Response exceeds the character limit.")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("Return one JSON object with no Markdown or duplicate keys.") from exc
    allowed = ("ask", "instruct") if role == "expert" else ("reply", "unclear", "move", "toggle")
    if role == "operator" and protocol_version in ("v4", "v6"):
        allowed += ("satisfied",)
    if not isinstance(data, dict) or data.get("type") not in allowed:
        raise ProtocolError("Unknown message type.")
    kind = data["type"]
    key = "target" if kind in ("move", "toggle") else "text"
    if set(data) != {"type", key} or not isinstance(data[key], str) or not data[key].strip():
        raise ProtocolError(f"Expected only type and a nonempty {key} string.")
    if key == "target" and len(data[key]) > 64:
        raise ProtocolError("Target ID is too long.")
    if operator_turn is not None:
        responses = operator_responses(operator_turn, protocol_version)
        if kind not in responses:
            code = "instruction_following_failure" if operator_turn == "instruct" else "invalid_output"
            raise ProtocolError(
                f"An {operator_turn} turn permits only {', '.join(responses)}.",
                code=code,
            )
        if kind in TARGET_PATTERNS and re.fullmatch(TARGET_PATTERNS[kind], data[key]) is None:
            raise ProtocolError(f"Invalid {kind} argument: target must match {TARGET_PATTERNS[kind]}.")
    return Message(kind, data[key])
