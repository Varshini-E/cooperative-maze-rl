import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from panel_guide.conversation import Completion, Conversation, Recorder
from panel_guide.protocol import ProtocolError, parse_message
from panel_guide.scripted import ScriptedExpert, ScriptedOperator
from panel_guide.tasks import State, example_task
from panel_guide.tinker_backend import model_messages


class QueueAgent:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, Completion) else Completion(value)


ASK = '{"type":"ask","text":"Describe the room."}'
INSTRUCT = '{"type":"instruct","text":"Take the bars door."}'
MOVE = '{"type":"move","target":"door_2"}'


class ProtocolTests(unittest.TestCase):
    def test_ambiguous_and_malformed_envelopes_are_rejected(self):
        bad = ['[]', '{}', 'null', '```json\n{}\n```',
               '{"type":"ask","type":"instruct","text":"go"}',
               '{"type":"ask","text":"go","target":"door_1"}',
               '{"type":"ask","text":5}', '{"type":"ask","text":" "}',
               '{"type":"move","target":"door_1"}', 'x' * 4097]
        for value in bad:
            with self.subTest(value=value[:80]), self.assertRaises(ProtocolError):
                parse_message(value, "expert")


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixture_succeeds_and_records_separate_audit(self):
        conversation = Conversation(example_task(), max_actions=9, max_exchanges=18)
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory))
            result = await conversation.run(ScriptedExpert(), ScriptedOperator(), recorder)
            self.assertEqual((result.status, result.actions, result.exchanges, result.reward),
                             ("success", 9, 18, 1.0))
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            requests = [e for e in events if e["type"] == "request"]
            self.assertEqual(len(requests), 36)
            self.assertEqual(len(list(Path(directory).glob("maze-*.png"))), 18)
            self.assertEqual(json.loads((Path(directory) / "summary.json").read_text())["reward"], 1)
            for request in requests:
                serialized = json.dumps(request)
                self.assertNotIn('"audit_state"', serialized)
                self.assertNotIn('"destination"', serialized)
                if request["role"] == "expert":
                    self.assertIsNone(request["image"])
                    self.assertIn("PRIVATE MANUAL", serialized)
                else:
                    self.assertIsInstance(request["image"], str)
                    self.assertNotIn("PRIVATE MANUAL", serialized)
                    self.assertNotIn("Goal: reach Vault", serialized)
            self.assertTrue((Path(directory) / "audit.jsonl").exists())
        with self.assertRaises(RuntimeError):
            await conversation.run(ScriptedExpert(), ScriptedOperator())

    async def test_slot_shuffle_does_not_break_symbol_grounding(self):
        task = example_task()
        rooms = tuple(replace(room, doors=tuple(replace(door, id=f"door_{i}")
            for i, door in enumerate(reversed(room.doors), 1))) for room in task.rooms)
        result = await Conversation(replace(task, rooms=rooms)).run(ScriptedExpert(), ScriptedOperator())
        self.assertEqual((result.status, result.actions), ("success", 9))

    async def test_already_on_switch_is_not_toggled_off(self):
        task = replace(example_task(), initial=State("Crescent", (True, True)))
        result = await Conversation(task).run(ScriptedExpert(), ScriptedOperator())
        self.assertEqual((result.status, result.actions), ("success", 7))

    async def test_ask_cannot_execute_an_action(self):
        conversation = Conversation(example_task(), max_exchanges=1)
        result = await conversation.run(QueueAgent(ASK), QueueAgent(MOVE))
        self.assertEqual((result.reason, result.actions, result.exchanges), ("exchange_limit", 0, 1))
        self.assertEqual(conversation.game.audit_state, example_task().initial)
        self.assertEqual(conversation.history[-1]["error"], "invalid_output")

    async def test_invalid_expert_skips_operator_and_consumes_exchange(self):
        operator = QueueAgent()
        conversation = Conversation(example_task(), max_exchanges=2)
        result = await conversation.run(QueueAgent("bad JSON", Completion(INSTRUCT, "length")), operator)
        self.assertEqual((result.exchanges, result.actions), (2, 0))
        self.assertFalse(operator.requests)

    async def test_malformed_operator_and_clarification_do_not_mutate_state(self):
        conversation = Conversation(example_task(), max_exchanges=3)
        result = await conversation.run(QueueAgent(INSTRUCT, INSTRUCT, INSTRUCT),
            QueueAgent('not JSON', '{"type":"unclear","text":"Which door?"}', MOVE))
        self.assertEqual(result.actions, 1)
        self.assertEqual(conversation.game.audit_state.room, "Workshop")

    async def test_invalid_target_consumes_action_budget(self):
        conversation = Conversation(example_task(), max_actions=1)
        result = await conversation.run(QueueAgent(INSTRUCT), QueueAgent('{"type":"move","target":"door_99"}'))
        self.assertEqual((result.reason, result.actions, result.reward), ("action_limit", 1, 0))

    async def test_provider_failure_is_interrupted_with_no_action_or_retry(self):
        operator = QueueAgent(ConnectionError("do not persist credentials from exceptions"))
        conversation = Conversation(example_task())
        result = await conversation.run(QueueAgent(INSTRUCT), operator)
        self.assertEqual((result.status, result.reason, result.actions),
                         ("interrupted", "infrastructure_error", 0))
        self.assertEqual(len(operator.requests), 1)

    async def test_timeout_does_not_execute_a_late_action(self):
        class SlowOperator:
            async def complete(self, request):
                await asyncio.sleep(1)
                return Completion(MOVE)
        result = await Conversation(example_task()).run(QueueAgent(INSTRUCT), SlowOperator(), timeout_seconds=0.01)
        self.assertEqual((result.reason, result.actions), ("infrastructure_error", 0))

    async def test_real_operator_request_contains_only_one_current_image(self):
        expert = QueueAgent(INSTRUCT, ASK)
        operator = QueueAgent(MOVE, '{"type":"reply","text":"anvil"}')
        await Conversation(example_task(), max_exchanges=2).run(expert, operator)
        self.assertNotEqual(operator.requests[0].image_png, operator.requests[1].image_png)
        for request in expert.requests:
            self.assertIsNone(request.image_png)
            self.assertIsNone(request.scripted_view)
            self.assertTrue(all(isinstance(m["content"], str) for m in model_messages(request)))
        for request in operator.requests:
            messages = model_messages(request)
            parts = messages[-1]["content"]
            self.assertEqual([part["type"] for part in parts], ["image", "text"])
            self.assertNotIn("scripted_view", parts[1]["text"])
            self.assertNotIn("MAZE MANUAL", str(messages))


if __name__ == "__main__":
    unittest.main()
