"""v9 private Expert notes."""
import asyncio
import json
import unittest

from panel_guide.conversation import Completion, Conversation
from panel_guide.curriculum import development_task
from panel_guide.protocol import NOTES_LIMIT, ProtocolError, parse_message


class ScriptedExpert:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    async def complete(self, request):
        self.requests.append(request)
        return Completion(json.dumps(self.replies.pop(0)))


class RecordingOperator:
    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return Completion(json.dumps({"type": "reply", "text": "Room emblem: anvil."}))


def play(replies, prompt_version="v9", exchanges=None):
    task = development_task("one-switch", 0)
    expert, operator = ScriptedExpert(replies), RecordingOperator()
    game = Conversation(task, prompt_version=prompt_version, max_exchanges=exchanges or len(replies))
    result = asyncio.run(game.run(expert, operator))
    return game, expert, operator, result


def user_payload(request):
    return json.loads(request.text_messages()[1]["content"])


ASK = {"type": "ask", "text": "What room are you in?"}


class NotesTests(unittest.TestCase):
    def test_parser_accepts_notes_only_when_enabled(self):
        raw = json.dumps({**ASK, "notes": "Room 5."})
        self.assertEqual(parse_message(raw, "expert", allow_notes=True).notes, "Room 5.")
        with self.assertRaises(ProtocolError):
            parse_message(raw, "expert")
        with self.assertRaises(ProtocolError):
            parse_message(json.dumps({**ASK, "notes": "x" * (NOTES_LIMIT + 1)}), "expert", allow_notes=True)
        with self.assertRaises(ProtocolError):
            parse_message(json.dumps({"type": "reply", "text": "hi", "notes": "n"}), "operator", allow_notes=True)

    def test_notes_are_shown_back_replaced_and_kept_when_omitted(self):
        _, expert, _, _ = play([{**ASK, "notes": "first"}, {**ASK, "notes": "second"}, ASK, ASK])
        seen = [user_payload(r)["your_private_notes"] for r in expert.requests]
        self.assertEqual(seen, ["(none yet)", "first", "second", "second"])

    def test_operator_never_sees_notes(self):
        game, _, operator, _ = play([{**ASK, "notes": "SECRET-NOTE"}, ASK, ASK])
        for request in operator.requests:
            self.assertIsNone(request.expert_notes)
            self.assertNotIn("SECRET-NOTE", json.dumps(request.text_messages()))
            self.assertNotIn("PRIVATE NOTES", request.system)
        self.assertNotIn("notes", json.dumps(game.history))

    def test_overlong_notes_end_the_episode(self):
        _, _, operator, result = play([{**ASK, "notes": "x" * (NOTES_LIMIT + 1)}, ASK])
        self.assertEqual((result.status, result.reason), ("failure", "protocol_failure"))
        self.assertEqual(operator.requests, [])

    def test_v8_still_rejects_notes_and_has_no_notes_field(self):
        _, expert, _, result = play([{**ASK, "notes": "n"}, ASK], prompt_version="v8")
        self.assertEqual(result.reason, "protocol_failure")
        self.assertNotIn("your_private_notes", user_payload(expert.requests[0]))


if __name__ == "__main__":
    unittest.main()
