import json
import tempfile
import unittest
from pathlib import Path

from panel_guide.conversation import Completion, Conversation, Recorder
from panel_guide.protocol import ProtocolError, operator_response_schema, parse_message
from panel_guide.scripted import ScriptedExpert, ScriptedOperator
from panel_guide.tasks import example_task
from panel_guide.tinker_backend import model_messages


class QueueAgent:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        return response if isinstance(response, Completion) else Completion(response)


ASK = '{"type":"ask","text":"Describe the doors."}'
INSTRUCT = '{"type":"instruct","text":"Use the requested local door."}'
MOVE = '{"type":"move","target":"door_1"}'


class ToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_turn_reply_is_rejected_and_next_turn_can_recover(self):
        expert = QueueAgent(INSTRUCT, INSTRUCT)
        operator = QueueAgent('{"type":"reply","text":"The ring door is open."}', MOVE)
        conversation = Conversation(example_task(), max_exchanges=2, prompt_version="v3")
        with tempfile.TemporaryDirectory() as directory:
            result = await conversation.run(expert, operator, Recorder(Path(directory)))
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
        self.assertEqual((result.actions, result.exchanges), (1, 2))
        self.assertEqual(conversation.game.audit_state.room, "Garden")
        self.assertEqual(len(operator.requests), 2)  # No hidden repair calls.
        self.assertFalse(any(e["type"] == "accepted_message" and e["role"] == "operator"
                             and e["exchange"] == 1 for e in events))
        self.assertTrue(any(e["type"] == "completion" and e["role"] == "operator"
                            and e["exchange"] == 1 for e in events))
        feedback = [e for e in events if e["type"] == "feedback"][0]
        self.assertEqual(feedback["error"], "instruction_following_failure")
        self.assertFalse(feedback["executed"])
        next_input = expert.requests[1].text_messages()[-1]["content"]
        self.assertIn("No action was executed", next_input)
        self.assertIn("instruction_following_failure", next_input)

    async def test_ask_cannot_dispatch_a_tool(self):
        conversation = Conversation(example_task(), max_exchanges=1, prompt_version="v3")
        result = await conversation.run(QueueAgent(ASK), QueueAgent(MOVE))
        self.assertEqual(result.actions, 0)
        self.assertEqual(conversation.game.audit_state, example_task().initial)
        self.assertEqual(conversation.history[-1]["error"], "invalid_output")
        self.assertFalse(conversation.history[-1]["executed"])

    async def test_malformed_arguments_multiple_calls_and_truncation_cannot_execute(self):
        responses = (
            '{"type":"move","target":"control_1"}',
            '{"type":"toggle","target":"door_1"}',
            '{"type":"move","target":"door_0"}',
            '{"type":"move","target":"door_1\\n"}',
            '{"type":"move","target":"door_1","extra":"ignored?"}',
            '[' + MOVE + ',' + MOVE + ']',
            MOVE + '\n' + MOVE,
            Completion(MOVE, "length"),
        )
        for response in responses:
            with self.subTest(response=response):
                conversation = Conversation(example_task(), max_exchanges=1, prompt_version="v3")
                result = await conversation.run(QueueAgent(INSTRUCT), QueueAgent(response))
                self.assertEqual(result.actions, 0)
                self.assertEqual(conversation.game.audit_state, example_task().initial)
                self.assertEqual(conversation.history[-1]["error"], "invalid_output")

    async def test_tool_receipts_distinguish_success_blocked_and_invalid_target(self):
        expert = QueueAgent(INSTRUCT, INSTRUCT, INSTRUCT)
        operator = QueueAgent(MOVE, '{"type":"move","target":"door_2"}',
                              '{"type":"move","target":"door_99"}')
        conversation = Conversation(example_task(), max_actions=3, prompt_version="v3")
        with tempfile.TemporaryDirectory() as directory:
            result = await conversation.run(expert, operator, Recorder(Path(directory)))
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
        self.assertEqual((result.reason, result.actions), ("action_limit", 3))
        self.assertEqual(conversation.game.audit_state.room, "Garden")
        calls = [e for e in events if e["type"] == "tool_call"]
        receipts = [e for e in events if e["type"] == "feedback"]
        self.assertEqual(len(calls), 3)
        self.assertEqual([c["tool_call_id"] for c in calls], [r["tool_call_id"] for r in receipts])
        self.assertEqual(len({r["tool_call_id"] for r in receipts}), 3)
        self.assertEqual([r["outcome"] for r in receipts], ["moved", "blocked", "invalid_target"])
        self.assertEqual([r["executed"] for r in receipts], [True, False, False])
        for receipt in receipts:
            self.assertFalse({"room", "switches", "destination", "emblem"} & receipt.keys())
        self.assertIn('"executed": false', expert.requests[2].text_messages()[-1]["content"])
        self.assertIn('"outcome": "blocked"', expert.requests[2].text_messages()[-1]["content"])

    async def test_clarification_is_a_valid_non_action(self):
        conversation = Conversation(example_task(), max_exchanges=1, prompt_version="v3")
        result = await conversation.run(QueueAgent(INSTRUCT),
            QueueAgent('{"type":"unclear","text":"No controls are visible."}'))
        self.assertEqual(result.actions, 0)
        self.assertNotIn("error", conversation.history[-1])
        self.assertNotIn("tool_call_id", conversation.history[-1])
        self.assertFalse(conversation.history[-1]["executed"])

    async def test_current_schema_is_turn_specific_and_contains_no_room_state(self):
        conversation = Conversation(example_task(), prompt_version="v3")
        for turn, allowed in (("ask", {"reply", "unclear"}), ("instruct", {"move", "toggle", "unclear"})):
            conversation.history = [{"role": "expert", "type": turn, "text": "Current message"}]
            messages = model_messages(conversation._request("operator"))
            parts = messages[-1]["content"]
            self.assertEqual([p["type"] for p in parts], ["image", "text"])
            schema = json.loads(parts[1]["text"].split("CURRENT TURN CONTRACT (JSON Schema):\n")[1])
            self.assertEqual({s["properties"]["type"]["const"] for s in schema["oneOf"]}, allowed)
            self.assertNotIn("PRIVATE MANUAL", str(messages))
            self.assertNotIn("scripted_view", str(messages))
            self.assertFalse(any(word in json.dumps(schema) for word in ("destination", "emblem", "is_open")))
        # Schemas are fresh values, not mutable shared state across episodes.
        operator_response_schema("ask")["oneOf"].clear()
        self.assertEqual(len(operator_response_schema("ask")["oneOf"]), 2)
        with self.assertRaises(ProtocolError):
            parse_message('{"type":"reply","text":"Description"}', "operator", operator_turn="instruct")

    async def test_scripted_episode_succeeds_with_strict_contract(self):
        result = await Conversation(example_task(), prompt_version="v3").run(ScriptedExpert(), ScriptedOperator())
        self.assertEqual((result.status, result.actions, result.exchanges, result.reward), ("success", 9, 18, 1.0))


if __name__ == "__main__":
    unittest.main()
