import json
import tomllib
import unittest
from pathlib import Path

from panel_guide.conversation import Completion, Conversation, OPERATOR_PROMPT_V7
from panel_guide.tasks import example_task
from panel_guide.tinker_backend import model_messages
from tests.test_tools import QueueAgent


def message(kind, text):
    return json.dumps({"type": kind, "text": text})


class CommunicationTests(unittest.IsolatedAsyncioTestCase):
    def test_current_config_and_prompt_examples(self):
        config = tomllib.loads(Path("configs/maze.toml").read_text())
        self.assertEqual(config["prompt_version"], "v7")
        self.assertEqual(config["operator_history"], "recent2")
        self.assertEqual(config["observation_mode"], "maze_detail")
        examples = [json.loads(line) for line in OPERATOR_PROMPT_V7.splitlines() if line.startswith('{')]
        self.assertTrue(any(e['type'] == 'reply' and 'LOCKED' in e['text'] for e in examples))
        self.assertTrue(any(e['type'] == 'reply' and 'identify' in e['text'] for e in examples))
        self.assertFalse(any(e.get('text') == 'observations' for e in examples))

    async def test_recent_two_complete_exchanges_keep_receipts_but_not_old_context_or_images(self):
        expert = QueueAgent(message("ask", "Old question: OLD_CONTEXT_MARKER"),
                            message("instruct", "Move through door_1."),
                            message("instruct", "Move through door_2."),
                            message("ask", "What is visible now?"))
        operator = QueueAgent(message("reply", "OLD_REPORT_MARKER"),
                              '{"type":"move","target":"door_1"}',
                              '{"type":"move","target":"door_2"}',
                              message("reply", "Current room: leaf."))
        conversation = Conversation(example_task(), prompt_version="v7", operator_history="recent2",
                                    observation_mode="maze_detail", max_exchanges=4)
        result = await conversation.run(expert, operator)
        self.assertEqual((result.exchanges, result.actions), (4, 2))
        first, moved_from, after_move, after_block = operator.requests
        self.assertEqual(len(first.history), 1)
        self.assertEqual(len(moved_from.history), 4)
        self.assertEqual(len(after_move.history), 7)
        self.assertEqual(len(after_block.history), 7)
        receipts = [e for e in after_block.history if e['role'] == 'environment']
        self.assertEqual([e['outcome'] for e in receipts], ['moved', 'blocked'])
        self.assertEqual([e['executed'] for e in receipts], [True, False])
        self.assertEqual([e['tool_call_id'] for e in receipts], ['exchange-2', 'exchange-3'])
        payload = model_messages(after_block)
        self.assertNotIn('OLD_CONTEXT_MARKER', str(payload))
        self.assertNotIn('OLD_REPORT_MARKER', str(payload))
        self.assertNotIn('PRIVATE MANUAL', str(payload))
        self.assertNotIn('Goal: reach Vault', str(payload))
        self.assertNotIn('scripted_view', str(payload))
        self.assertEqual(sum(p['type'] == 'image' for p in payload[-1]['content']), 2)
        self.assertTrue(all(isinstance(m['content'], str) for m in payload[:-1]))
        self.assertNotEqual(moved_from.image_png, after_move.image_png)
        self.assertEqual(after_move.image_png, after_block.image_png)
        self.assertEqual(after_move.detail_png, after_block.detail_png)
        self.assertIn('OLD_REPORT_MARKER', str(expert.requests[-1].history))
        after_block.history[0]['text'] = 'mutated copy'
        self.assertNotIn('mutated copy', str(conversation.history))

    async def test_followup_sees_previous_clarification_and_no_action_receipt(self):
        operator = QueueAgent(message('unclear', 'Which door do you mean?'),
                              '{"type":"move","target":"door_1"}')
        conversation = Conversation(example_task(), prompt_version='v7', operator_history='recent2',
                                    max_exchanges=2)
        await conversation.run(QueueAgent(message('instruct', 'Use that door.'),
                                          message('instruct', 'I mean door_1.')), operator)
        recent = operator.requests[1].history
        self.assertEqual([e['role'] for e in recent], ['expert', 'operator', 'environment', 'expert'])
        self.assertEqual(recent[1]['text'], 'Which door do you mean?')
        self.assertFalse(recent[2]['executed'])
        self.assertEqual(recent[3]['text'], 'I mean door_1.')

    async def test_rejected_outputs_still_bound_history_by_complete_exchanges(self):
        operator = QueueAgent(message('reply', 'OLD_REPORT'), 'not JSON', message('reply', 'Current leaf.'))
        conversation = Conversation(example_task(), prompt_version='v7', operator_history='recent2',
                                    max_exchanges=4)
        await conversation.run(QueueAgent(message('ask', 'Old question'), 'invalid Expert',
                                          message('ask', 'Second question'), message('ask', 'Current question')), operator)
        recent = operator.requests[-1].history
        self.assertNotIn('OLD_REPORT', str(recent))
        receipts = [e for e in recent if e['role'] == 'environment']
        self.assertEqual([e['sender'] for e in receipts], ['expert', 'operator'])
        self.assertTrue(all(e['error'] == 'invalid_output' and not e['executed'] for e in receipts))

    async def test_mislabeled_question_is_delivered_as_unverified_report(self):
        expert = QueueAgent(message('instruct', 'What is the current room emblem?'),
                            message('ask', 'Which doors are open?'))
        operator = QueueAgent(message('reply', 'Current emblem: crescent.'),
                              message('reply', 'Both local doors are OPEN.'))
        conversation = Conversation(example_task(), prompt_version='v7', operator_history='recent2',
                                    max_exchanges=2)
        result = await conversation.run(expert, operator)
        self.assertEqual((result.actions, result.reward), (0, 0))
        self.assertIn('Current emblem: crescent.', str(expert.requests[1].history))
        receipt = next(e for e in operator.requests[1].history if e['role'] == 'environment')
        self.assertFalse(receipt['observation_verified'])
        self.assertFalse(receipt['executed'])
        self.assertNotIn('error', receipt)

    async def test_action_boundaries_remain_strict_in_v7(self):
        for kind, response in [('ask', '{"type":"move","target":"door_1"}'),
                               ('instruct', message('reply', 'I moved to the goal.')),
                               ('instruct', '{"type":"move","target":"control_1"}'),
                               ('instruct', Completion('{"type":"move","target":"door_1"}', 'length'))]:
            with self.subTest(kind=kind, response=response):
                conversation = Conversation(example_task(), prompt_version='v7', operator_history='recent2',
                                            max_exchanges=1)
                result = await conversation.run(QueueAgent(message(kind, 'Go to the goal.')), QueueAgent(response))
                self.assertEqual((result.actions, result.reward), (0, 0))
                self.assertEqual(conversation.game.audit_state, example_task().initial)


if __name__ == '__main__':
    unittest.main()
