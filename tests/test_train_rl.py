"""Training loop pieces with fake models and a fake Tinker client (no network)."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from panel_guide import train_rl
    from tests.test_env import ACTION, MOVE, Agent, Policy, Renderer
    HAS_COOKBOOK = True
except ImportError:
    HAS_COOKBOOK = False

from panel_guide.conversation import BudgetExceeded
from panel_guide.curriculum import development_task
from panel_guide.tinker_backend import SpendingGuard


def rows():
    return [{"task_id": f"t{m}-{n}", "maze_id": f"m{m}", "split": split, "solution_length": n}
            for m in range(4) for n in range(1, 11) for split in ("train", "validation_mazes")]


class Future:
    def __init__(self, value):
        self.value = value

    async def result_async(self):
        return self.value


class FakeClient:
    def __init__(self):
        self.calls = []

    async def forward_backward_async(self, data, loss_fn):
        self.calls.append(("forward_backward", loss_fn))
        return Future(SimpleNamespace(metrics={"loss:sum": 0.5}))

    async def optim_step_async(self, params):
        self.calls.append(("optim_step", params.learning_rate))
        return Future(SimpleNamespace(metrics={"grad_norm": 0.1}))

    async def save_state_async(self, name, ttl_seconds):
        return Future(SimpleNamespace(path=f"tinker://state/{name}"))

    async def save_weights_for_sampler_async(self, name, ttl_seconds):
        return Future(SimpleNamespace(path=f"tinker://sampler/{name}"))


class FakeService:
    async def create_sampling_client_async(self, model_path, retry_config):
        return SimpleNamespace(path=model_path)


@unittest.skipUnless(HAS_COOKBOOK, "Install optional models dependencies to test the training loop.")
class TrainLoopTests(unittest.IsolatedAsyncioTestCase):
    def test_pool_keeps_requested_split_and_routes(self):
        pool = train_rl.training_pool(rows(), "train", (5, 10))
        self.assertEqual(len(pool), 24)
        self.assertTrue(all(r["split"] == "train" and r["solution_length"] >= 5 for r in pool))
        with self.assertRaises(ValueError):
            train_rl.training_pool(rows(), "test_mazes")

    def test_budget_stop_is_found_through_exception_chain(self):
        try:
            try:
                raise BudgetExceeded("cap")
            except BudgetExceeded as exc:
                raise RuntimeError("wrapped") from exc
        except RuntimeError as exc:
            self.assertTrue(train_rl.is_budget_stop(exc))
        self.assertTrue(train_rl.is_budget_stop(RuntimeError("spending_limit")))
        self.assertFalse(train_rl.is_budget_stop(TimeoutError()))

    async def test_group_runs_fresh_episodes_and_saves_trajectories(self):
        task = development_task("one-door")
        row = {"task": asdict(task), "task_id": "one-door", "max_actions": 4, "max_exchanges": 1}
        with tempfile.TemporaryDirectory() as directory:
            group, errors = await train_rl.collect_group(
                row, "v9", Policy(MOVE, MOVE), Agent(ACTION, ACTION), Renderer(), Path(directory),
                group_size=2, max_tokens=512, context_tokens=32768, timeout_seconds=5)
            self.assertEqual((errors, group.get_total_rewards()), ([], [1.0, 1.0]))
            self.assertEqual(len(list(Path(directory).glob("attempt-*/trajectory.json"))), 2)

    async def test_interrupted_episode_drops_the_whole_group(self):
        row = {"task": asdict(development_task("one-door")), "task_id": "one-door", "max_actions": 4, "max_exchanges": 1}
        with tempfile.TemporaryDirectory() as directory:
            group, errors = await train_rl.collect_group(
                row, "v9", Policy(MOVE, MOVE), Agent(ACTION, ConnectionError("down")), Renderer(), Path(directory),
                group_size=2, max_tokens=512, context_tokens=32768, timeout_seconds=5)
        self.assertIsNone(group)
        self.assertEqual(len(errors), 1)

    async def test_update_reserves_steps_and_reloads_sampler(self):
        settings = {"train_usd_per_million": 0.44, "loss_fn": "importance_sampling", "learning_rate": 1e-4,
                    "grad_clip_norm": 1.0, "checkpoint_ttl_seconds": 60}
        expert = SimpleNamespace(config=SimpleNamespace(model="Qwen/Qwen3-8B"), sampler=None,
                                 model_path=None, policy_id=None)
        client, guard = FakeClient(), SpendingGuard(1.0)
        data = [SimpleNamespace(model_input=SimpleNamespace(length=1000))]
        checkpoint, metrics = await train_rl.update_policy(client, FakeService(), expert, guard, data, settings, "step-001")
        self.assertEqual(client.calls, [("forward_backward", "importance_sampling"), ("optim_step", 1e-4)])
        self.assertEqual(expert.model_path, "tinker://sampler/step-001")
        self.assertEqual(checkpoint["state_path"], "tinker://state/step-001")
        self.assertEqual(set(metrics), {"loss:sum", "grad_norm"})
        self.assertGreater(guard.reserved_usd, 0)

    async def test_full_loop_trains_on_mixed_groups_and_records_batches(self):
        task = asdict(development_task("one-door"))
        bank = [{"task": task, "task_id": f"one-door-{m}", "maze_id": f"m{m}", "split": "train",
                 "solution_length": 5, "max_actions": 4, "max_exchanges": 1} for m in range(2)]
        unclear = '{"type":"unclear","text":"Which door?"}'
        expert = SimpleNamespace(config=SimpleNamespace(model="Qwen/Qwen3-8B", max_tokens=512, context_tokens=32768),
                                 renderer=Renderer(), sampler=None, model_path=None, policy_id=None)
        client, service = FakeClient(), FakeService()
        service.create_lora_training_client_async = lambda **kwargs: Future(client).result_async()
        service.close = lambda status: Future(None).result_async()
        operator = Agent(*[ACTION, unclear] * 4)  # one success and one failure per group
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "maze.toml"
            config.write_text(Path("configs/maze.toml").read_text()
                              .replace("group_size = 8", "group_size = 2").replace("tasks_per_batch = 6", "tasks_per_batch = 2"))
            args = SimpleNamespace(config=config, bank=Path("unused"), split="train", min_route=5, max_route=10,
                                   batches=2, check=False, budget_usd=5.0, output_dir=Path(directory) / "run",
                                   seed=0, resume=None, parallel_groups=1)
            with patch.object(train_rl, "load_bank", return_value=({}, bank)), \
                    patch.dict("os.environ", {"TINKER_API_KEY": "test"}), \
                    patch("panel_guide.tinker_backend.create_agents", return_value=(expert, operator, service)), \
                    patch("panel_guide.tinker_env.ExpertTokenPolicy", return_value=Policy(*[MOVE] * 8)), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(await train_rl.run(args), 0)
            summary = (Path(directory) / "run/summary.json").read_text()
            batches = [e for e in map(json.loads, (Path(directory) / "run/trace.jsonl").read_text().splitlines())
                       if e["type"] == "batch"]
        self.assertIn('"complete"', summary)
        self.assertEqual([(b["successes"], b["episodes"], b["mixed_groups"]) for b in batches], [(2, 4, 2)] * 2)
        self.assertEqual([c[0] for c in client.calls], ["forward_backward", "optim_step"] * 2)
        self.assertEqual(expert.model_path, "tinker://sampler/step-002")

    async def test_check_is_free_and_paid_run_needs_a_budget(self):
        args = SimpleNamespace(config=Path("configs/maze.toml"), bank=Path("unused"), split="train",
                               min_route=5, max_route=10, batches=6, check=True, budget_usd=None)
        with patch.object(train_rl, "load_bank", return_value=({}, rows())), \
                patch("panel_guide.tinker_backend.create_agents", side_effect=AssertionError("network")):
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(await train_rl.run(args), 0)
            self.assertIn('"episodes": 288', out.getvalue())
            args.check = False
            with self.assertRaises(ValueError):
                await train_rl.run(args)


if __name__ == "__main__":
    unittest.main()
