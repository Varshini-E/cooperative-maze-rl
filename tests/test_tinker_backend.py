import unittest
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from panel_guide.conversation import BudgetExceeded, Conversation
from panel_guide.tasks import example_task
from panel_guide.tinker_backend import ModelConfig, SpendingGuard, TinkerAgent

CONFIG = ModelConfig("model", "renderer", 512, 32768, 0, 0.195, 0.60)


class FakeRenderer:
    def build_generation_prompt(self, messages):
        self.messages = messages
        return SimpleNamespace(length=100, to_ints=lambda: [1, 2, 3],
                               model_dump=lambda **kw: {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]})

    def get_stop_sequences(self):
        return [9]

    def parse_response(self, tokens):
        return ({"content": [{"type": "thinking", "thinking": "private reasoning"},
                             {"type": "text", "text": '{"type":"ask","text":"Where are you?"}'}]},
                SimpleNamespace(is_clean=True))


class FakeSampler:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("a provider error")
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[4, 5, 9],
            logprobs=[-0.1, -0.2, -0.3], stop_reason="stop")])


class BudgetTests(unittest.TestCase):
    def test_invalid_caps_fail_closed(self):
        for cap in (0, -1, float("inf"), float("nan")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                SpendingGuard(cap)

    def test_reservations_include_maximum_output_and_accumulate(self):
        guard = SpendingGuard(0.001)
        expected = 1.2 * (100 * 0.195 + 512 * 0.60) / 1_000_000
        self.assertAlmostEqual(guard.reserve("expert", CONFIG, 100), expected)
        guard.reserve("operator", CONFIG, 100)
        with self.assertRaises(BudgetExceeded):
            guard.reserve("expert", CONFIG, 100)
        self.assertEqual(len(guard.reservations), 2)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_sampler_payload_and_raw_output_are_captured_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = FakeSampler()
            agent = TinkerAgent(CONFIG, FakeRenderer(), sampler, SpendingGuard(1),
                lambda **kw: SimpleNamespace(model_dump=lambda **opts: kw),
                capture_dir=Path(directory), decode_tokens=lambda tokens: "PRIVATE raw generated output")
            completion = await agent.complete(Conversation(example_task())._request("expert"))
            payload = json.loads(Path(completion.sample["provider_request"]).read_text())
            self.assertEqual(payload["prompt"], sampler.calls[0]["prompt"].model_dump())
            self.assertEqual(payload["sampling_params"], sampler.calls[0]["sampling_params"].model_dump())
            self.assertEqual(completion.sample["raw_text"], "PRIVATE raw generated output")
            self.assertNotIn("PRIVATE", completion.text)

    async def test_adapter_preserves_tokens_but_does_not_forward_thinking(self):
        renderer, sampler, budget = FakeRenderer(), FakeSampler(), SpendingGuard(1)
        agent = TinkerAgent(CONFIG, renderer, sampler, budget, lambda **kwargs: kwargs)
        request = Conversation(example_task())._request("expert")
        completion = await agent.complete(request)
        self.assertNotIn("private reasoning", completion.text)
        self.assertEqual(completion.sample["token_ids"], [4, 5, 9])
        self.assertEqual(completion.sample["logprobs"], [-0.1, -0.2, -0.3])
        self.assertEqual(completion.sample["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(sampler.calls[0]["sampling_params"]["max_tokens"], 512)

    async def test_spending_limit_blocks_request_before_provider(self):
        sampler = FakeSampler()
        agent = TinkerAgent(CONFIG, FakeRenderer(), sampler, SpendingGuard(0.000001), lambda **kw: kw)
        result = await Conversation(example_task()).run(agent, None)
        self.assertEqual((result.status, result.reason, result.actions), ("interrupted", "spending_limit", 0))
        self.assertFalse(sampler.calls)

    async def test_failed_request_keeps_its_reservation_and_is_not_retried(self):
        sampler, budget = FakeSampler(fail=True), SpendingGuard(1)
        agent = TinkerAgent(CONFIG, FakeRenderer(), sampler, budget, lambda **kw: kw)
        result = await Conversation(example_task()).run(agent, None)
        self.assertEqual(result.reason, "infrastructure_error")
        self.assertGreater(budget.reserved_usd, 0)
        self.assertEqual(len(sampler.calls), 1)

    async def test_context_overflow_is_detected_before_reserving_or_sampling(self):
        sampler, budget = FakeSampler(), SpendingGuard(1)
        agent = TinkerAgent(replace(CONFIG, context_tokens=600), FakeRenderer(), sampler, budget, lambda **kw: kw)
        with self.assertRaises(ValueError):
            await agent.complete(Conversation(example_task())._request("expert"))
        self.assertFalse(sampler.calls)
        self.assertEqual(budget.reserved_usd, 0)

    async def test_bad_renderer_response_is_a_protocol_failure_and_keeps_sample(self):
        class MalformedRenderer(FakeRenderer):
            def parse_response(self, tokens):
                raise ValueError("Malformed sampled response")
        agent = TinkerAgent(CONFIG, MalformedRenderer(), FakeSampler(), SpendingGuard(1), lambda **kw: kw)
        completion = await agent.complete(Conversation(example_task())._request("expert"))
        self.assertEqual(completion.stop_reason, "invalid_termination")
        self.assertEqual(completion.sample["token_ids"], [4, 5, 9])
        result = await Conversation(example_task(), max_exchanges=1).run(agent, None)
        self.assertEqual((result.status, result.reason), ("failure", "exchange_limit"))


if __name__ == "__main__":
    unittest.main()
