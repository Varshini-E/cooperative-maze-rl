import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
    from tinker_cookbook.rl.rollouts import do_single_rollout
    from tinker_cookbook.rl.types import Env, TrajectoryGroup
    from panel_guide.tinker_env import ExpertTokenPolicy, MazeEnv, MazeGroupBuilder, RolloutInterrupted
    HAS_COOKBOOK = True
except ImportError:
    HAS_COOKBOOK = False

from panel_guide.conversation import BudgetExceeded, Completion, Conversation
from panel_guide.curriculum import development_task
from panel_guide.tasks import State
from panel_guide.tinker_backend import ModelConfig, SpendingGuard, TinkerAgent
from panel_guide.run_episode import run as run_episode

ASK = '{"type":"ask","text":"Describe the current room."}'
MOVE = '{"type":"instruct","text":"Move through the plus door."}'
REPLY = '{"type":"reply","text":"OPERATOR_ONLY_OBSERVATION: sun; plus door_3 open."}'
ACTION = '{"type":"move","target":"door_3"}'

class Renderer:
    """Small deterministic tokenizer for exact alignment tests, not model inference."""
    tokenizer = SimpleNamespace(decode=lambda tokens: bytes(t for t in tokens if t != 0).decode())

    def build_generation_prompt(self, messages):
        return tinker.ModelInput.from_ints(list(json.dumps(messages).encode()))

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, tokens):
        return {"role": "assistant", "content": self.tokenizer.decode(tokens)}, SimpleNamespace(is_clean=tokens[-1] == 0)

class Agent:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, Completion) else Completion(value)

class Policy:
    def __init__(self, *messages):
        self.messages = iter(messages)

    async def __call__(self, ob, stop, *, max_tokens=None):
        text = next(self.messages)
        tokens = list(text.encode()) + [0]
        return TokensWithLogprobs(tokens, [-.1 - i / 1000 for i in range(len(tokens))])

def make_env(operator, *, exchanges=2, task=None, recorder=None, **kwargs):
    conversation = Conversation(task or development_task("one-door"), prompt_version="v4",
        observation_mode="maze_detail", operator_history="latest", max_exchanges=exchanges, max_actions=4)
    return MazeEnv(conversation, operator, Renderer(), recorder=recorder, **kwargs)

@unittest.skipUnless(HAS_COOKBOOK, "Install optional models dependencies to test the native Env.")
class NativeEnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_v7_recent_history_reaches_operator_through_native_env_with_expert_only_targets(self):
        conversation = Conversation(development_task("one-door"), prompt_version="v7",
            observation_mode="maze_detail", operator_history="recent2", max_exchanges=4)
        operator = Agent(REPLY, REPLY, REPLY, ACTION)
        env = MazeEnv(conversation, operator, Renderer())
        trajectory = await do_single_rollout(Policy(ASK, ASK, ASK.replace('"ask"', '"instruct"'), MOVE), env)
        self.assertEqual([t.reward for t in trajectory.transitions], [0, 0, 0, 1])
        self.assertEqual([len(r.history) for r in operator.requests], [1, 4, 7, 7])
        self.assertEqual(len(operator.requests), 4)  # One Operator call per exchange.
        self.assertEqual(sum(e['role'] == 'environment' for e in operator.requests[-1].history), 2)
        self.assertIsNone(env.request.image_png)
        group = TrajectoryGroup([trajectory], [0.], [{}])
        data, _ = assemble_training_data([group], compute_advantages([group]))
        for datum in data:
            mask = datum.loss_fn_inputs['mask'].to_torch().tolist()
            targets = datum.loss_fn_inputs['target_tokens'].to_torch().tolist()
            trained = Renderer.tokenizer.decode([token for token, enabled in zip(targets, mask) if enabled])
            self.assertNotIn('OPERATOR_ONLY_OBSERVATION', trained)
            self.assertNotIn('PRIVATE MANUAL', trained)
            self.assertIn('"type":', trained)

    async def test_v6_inspection_reply_reaches_next_expert_observation_through_native_env(self):
        conversation = Conversation(development_task("one-door"), prompt_version="v6",
                                    observation_mode="maze_detail", operator_history="latest", max_exchanges=2)
        env = MazeEnv(conversation, Agent(REPLY, ACTION), Renderer())
        mislabeled_question = ASK.replace('"ask"', '"instruct"')
        trajectory = await do_single_rollout(Policy(mislabeled_question, MOVE), env)
        self.assertEqual([t.reward for t in trajectory.transitions], [0, 1])
        self.assertEqual(conversation.result.actions, 1)
        second = trajectory.transitions[1]
        self.assertIn('OPERATOR_ONLY_OBSERVATION', Renderer.tokenizer.decode(second.ob.to_ints()))
        self.assertNotIn('instruction_following_failure', Renderer.tokenizer.decode(second.ob.to_ints()))
        self.assertEqual(Renderer.tokenizer.decode(second.ac.tokens), MOVE)

    async def test_native_rollout_matches_conversation_requests_feedback_and_reward(self):
        direct_expert, direct_operator = Agent(ASK, MOVE), Agent(REPLY, ACTION)
        legacy = Conversation(development_task("one-door"), prompt_version="v4", observation_mode="maze_detail",
                              operator_history="latest", max_exchanges=2, max_actions=4)
        expected = await legacy.run(direct_expert, direct_operator)
        native_operator = Agent(REPLY, ACTION)
        env = make_env(native_operator)
        self.assertIsInstance(env, Env)
        trajectory = await do_single_rollout(Policy(ASK, MOVE), env)
        self.assertEqual(env.conversation.result, expected)
        self.assertEqual(env.conversation.history, legacy.history)
        for index, transition in enumerate(trajectory.transitions):
            prompt = Renderer().build_generation_prompt(direct_expert.requests[index].text_messages())
            self.assertEqual(transition.ob.to_ints(), prompt.to_ints())
            a, b = direct_operator.requests[index], native_operator.requests[index]
            self.assertEqual((a.text_messages(), a.image_png, a.detail_png), (b.text_messages(), b.image_png, b.detail_png))
            self.assertEqual(len(b.history), 1)
        self.assertEqual([t.reward for t in trajectory.transitions], [0, 1])
        self.assertEqual(trajectory.stop_reason, "completed")
        self.assertTrue(trajectory.transitions[-1].episode_done)
        with self.assertRaises(RuntimeError):
            await env.initial_observation()
        with self.assertRaises(RuntimeError):
            await env.step(list(MOVE.encode()) + [0])

    async def test_real_cookbook_datum_alignment_masks_and_group_advantages(self):
        success = await do_single_rollout(Policy(ASK, MOVE), make_env(Agent(REPLY, ACTION)))
        failure = await do_single_rollout(Policy(ASK, ASK), make_env(Agent(REPLY, REPLY)))
        group = TrajectoryGroup([success, failure], [0., 0.], [{}, {}])
        advantages = compute_advantages([group])
        self.assertEqual(advantages[0].tolist(), [.5, -.5])
        data, provenance = assemble_training_data([group], advantages)
        self.assertEqual(len(data), 4)  # Rebuilt v4 prompts are not prefix extensions.
        self.assertEqual(provenance, [{'group_idx': 0, 'traj_idx': i} for i in (0, 0, 1, 1)])
        offset = 0
        for trajectory, advantage in ((success, .5), (failure, -.5)):
            for transition in trajectory.transitions:
                datum = data[offset]
                offset += 1
                ob, action = transition.ob.to_ints(), transition.ac.tokens
                whole = ob + action
                fields = {k: v.to_torch().tolist() for k, v in datum.loss_fn_inputs.items()}
                self.assertEqual(datum.model_input.to_ints(), whole[:-1])
                self.assertEqual(fields['target_tokens'], whole[1:])
                self.assertEqual(fields['mask'], [0.] * (len(ob) - 1) + [1.] * len(action))
                self.assertEqual(fields['advantages'], [0.] * (len(ob) - 1) + [advantage] * len(action))
                self.assertTrue(all(p == 0 for p in fields['logprobs'][:len(ob) - 1]))
                for actual, expected in zip(fields['logprobs'][len(ob) - 1:], transition.ac.logprobs):
                    self.assertAlmostEqual(actual, expected, places=6)
                # Operator reports, manual, and replayed prior Expert messages are all observation tokens.
                targets = bytes(token for token, mask in zip(fields['target_tokens'], fields['mask']) if mask and token)
                self.assertNotIn(b'OPERATOR_ONLY_OBSERVATION', targets)
                self.assertNotIn(b'PRIVATE MANUAL', targets)

    async def test_truncation_invalid_json_and_satisfied_are_recoverable_zero_reward(self):
        operator = Agent('{"type":"satisfied","text":"The plus door is already open."}', ACTION)
        env = make_env(operator, exchanges=4)
        await env.initial_observation()
        first = await env.step(list(MOVE.encode()) + [0], extra={"stop_reason": "length"})
        second = await env.step(list(b'not JSON') + [0])
        third = await env.step(list(MOVE.encode()) + [0])
        self.assertEqual([first.reward, second.reward, third.reward], [0, 0, 0])
        self.assertFalse(any(s.episode_done for s in (first, second, third)))
        self.assertEqual(len(operator.requests), 1)
        self.assertEqual(env.conversation.game.remaining_actions, 4)
        last = await env.step(list(MOVE.encode()) + [0])
        self.assertEqual((last.reward, last.episode_done), (1, True))

    async def test_blocked_action_does_not_move_and_consumes_physical_budget(self):
        task = replace(development_task("full"), initial=State("Garden", (False, False)))
        env = make_env(Agent('{"type":"move","target":"door_2"}'), exchanges=1, task=task)
        trajectory = await do_single_rollout(Policy(MOVE), env)
        self.assertEqual(trajectory.transitions[0].reward, 0)
        self.assertEqual(env.conversation.game.audit_state, task.initial)
        self.assertEqual(env.conversation.result.actions, 1)
        self.assertEqual(env.conversation.history[-1]['outcome'], 'blocked')

    async def test_operator_failure_and_spending_interruptions_do_not_become_failed_samples(self):
        for error in (ConnectionError('private details'), BudgetExceeded('cap')):
            env = make_env(Agent(error))
            with self.assertRaises(RolloutInterrupted):
                await do_single_rollout(Policy(MOVE), env)
            self.assertEqual(env.conversation.result.status, 'interrupted')
            self.assertEqual(env.conversation.result.actions, 0)

    async def test_context_overflow_interrupts_before_sampling_or_operator_call(self):
        operator = Agent()
        env = make_env(operator, context_tokens=10, max_tokens=5)
        with self.assertRaises(RolloutInterrupted):
            await env.initial_observation()
        self.assertEqual(env.conversation.result.status, 'interrupted')
        self.assertFalse(operator.requests)

    async def test_group_members_reset_independently_and_reject_reused_or_mixed_tasks(self):
        builder = MazeGroupBuilder(lambda: make_env(Agent(ACTION)), 4)
        envs = await builder.make_envs()
        await do_single_rollout(Policy(MOVE), envs[0])
        self.assertTrue(envs[0].conversation.finished)
        for env in envs[1:]:
            self.assertEqual(env.conversation.exchanges, 0)
            self.assertEqual(env.conversation.history, [])
            self.assertEqual(env.conversation.game.audit_state, development_task('one-door').initial)
        same = make_env(Agent())
        with self.assertRaises(ValueError):
            await MazeGroupBuilder(lambda: same, 2).make_envs()
        mixed = iter((make_env(Agent()), make_env(Agent(), task=development_task('one-switch'))))
        with self.assertRaises(ValueError):
            await MazeGroupBuilder(lambda: next(mixed), 2).make_envs()

    async def test_policy_preserves_sampled_tokens_logprobs_and_reserves_before_sampling(self):
        tokens, logprobs = list(ASK.encode()) + [0], [-.25] * (len(ASK) + 1)
        sampler = SimpleNamespace(sample_async=AsyncMock(return_value=SimpleNamespace(sequences=[
            SimpleNamespace(tokens=tokens, logprobs=logprobs, stop_reason='stop')])))
        config = ModelConfig('test', 'test', 128, 32768, .3, .195, .6)
        budget = SpendingGuard(1)
        agent = TinkerAgent(config, Renderer(), sampler, budget, lambda **kwargs: kwargs)
        policy = ExpertTokenPolicy(agent)
        result = await policy(tinker.ModelInput.from_ints([10, 20]), [0], max_tokens=64)
        self.assertEqual(result.tokens, tokens)
        self.assertEqual(result.logprobs, logprobs)
        self.assertEqual(sampler.sample_async.call_args.kwargs['sampling_params']['max_tokens'], 64)
        self.assertEqual(budget.reservations[0]['max_output_tokens'], 64)
        limited = ExpertTokenPolicy(TinkerAgent(config, Renderer(), sampler, SpendingGuard(.00000001), lambda **kw: kw))
        with self.assertRaises(BudgetExceeded):
            await limited(tinker.ModelInput.from_ints([10]), [0])
        self.assertEqual(sampler.sample_async.await_count, 1)

    async def test_missing_or_invalid_logprobs_reject_rollout_after_preserving_cost(self):
        for logprobs in (None, [], [float('nan')]):
            sampler = SimpleNamespace(sample_async=AsyncMock(return_value=SimpleNamespace(sequences=[
                SimpleNamespace(tokens=[1], logprobs=logprobs, stop_reason='stop')])))
            budget = SpendingGuard(1)
            policy = ExpertTokenPolicy(TinkerAgent(ModelConfig('x', 'x', 32, 32768, .3, .2, .6),
                                       Renderer(), sampler, budget, lambda **kwargs: kwargs))
            with self.assertRaises(RolloutInterrupted):
                await policy(tinker.ModelInput.from_ints([1]), [0])
            self.assertGreater(budget.reserved_usd, 0)

    def sampled_expert(self, *messages):
        values = iter(messages)

        async def sample(**kwargs):
            tokens = list(next(values).encode()) + [0]
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, logprobs=[-.25] * len(tokens), stop_reason='stop')])

        return TinkerAgent(ModelConfig('test', 'test', 512, 32768, .3, .195, .6), Renderer(),
                           SimpleNamespace(sample_async=sample), SpendingGuard(1), lambda **kwargs: kwargs)

    async def test_single_episode_defaults_to_native_runner_and_saves_exact_trajectory(self):
        expert, operator = self.sampled_expert(ASK, MOVE), Agent(REPLY, ACTION)
        service = SimpleNamespace(close=AsyncMock())
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'TINKER_API_KEY': 'offline-test'}), \
                patch('panel_guide.run_episode.create_agents', new_callable=AsyncMock, return_value=(expert, operator, service)):
            args = SimpleNamespace(config=Path('configs/maze.toml'), check=False, backend='tinker', task='one-door',
                layout=0, partner='visual', runner=None, budget_usd=.1, output_dir=Path(folder))
            self.assertEqual(await run_episode(args), 0)
            manifest = json.loads((Path(folder) / 'run.json').read_text())
            trajectory = json.loads((Path(folder) / 'trajectory.json').read_text())
            self.assertEqual(manifest['runner'], 'cookbook')
            self.assertEqual([t['reward'] for t in trajectory['transitions']], [0, 1])
            self.assertEqual(trajectory['transitions'][0]['action_tokens'], list(ASK.encode()) + [0])
            service.close.assert_awaited_once_with('success')

if __name__ == '__main__':
    unittest.main()
