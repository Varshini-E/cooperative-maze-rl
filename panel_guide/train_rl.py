"""Train the Expert with a simplified GRPO-style loop on Tinker.

Each batch picks `tasks_per_batch` training tasks, runs `group_size` independent episodes on each,
applies one importance-sampling update from group-centred advantages and reloads the sampler with
the new weights. No PPO clipping or reference KL. Settings: `[training]` in configs/maze.toml.

    python -m panel_guide.train_rl --bank data/mazes-v2 --check            # free: plan and rough cost
    python -m panel_guide.train_rl --bank data/mazes-v2 --budget-usd 40    # paid
"""

import argparse
import asyncio
import json
import math
import os
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from panel_guide.conversation import BudgetExceeded, Conversation, Recorder
from panel_guide.maze_bank import load_bank
from panel_guide.task_sampler import TaskSampler
from panel_guide.tasks import task_from_dict

# Rough planning numbers from past runs (thinking on): sampling dominates the cost.
USD_PER_EPISODE, USD_PER_UPDATE = 0.13, 1.0


def training_pool(records, split="train", routes=(5, 10)):
    """Short routes are almost always solved, so their groups carry no signal."""
    low, high = routes
    pool = [r for r in records if r["split"] == split and low <= r["solution_length"] <= high]
    if not pool:
        raise ValueError(f"No {split} tasks with {low}-{high} action routes.")
    return pool


def plan_summary(pool, settings, batches):
    episodes = batches * settings["tasks_per_batch"] * settings["group_size"]
    return {"pool_tasks": len(pool), "pool_mazes": len({r["maze_id"] for r in pool}),
            "batches": batches, "episodes": episodes,
            "rough_cost_usd": round(episodes * USD_PER_EPISODE + batches * USD_PER_UPDATE, 2)}


def is_budget_stop(error):
    while error is not None:
        if isinstance(error, BudgetExceeded) or str(error) == "spending_limit":
            return True
        error = error.__cause__ or error.__context__
    return False


async def collect_group(row, prompt_version, policy, operator, renderer, output, *, group_size,
                        max_tokens, context_tokens, timeout_seconds):
    """Run one group; return (TrajectoryGroup or None if any episode was interrupted, errors)."""
    from tinker_cookbook.rl.rollouts import do_single_rollout
    from tinker_cookbook.rl.types import TrajectoryGroup

    from panel_guide.tinker_env import MazeEnv, MazeGroupBuilder, save_trajectory

    task = task_from_dict(row["task"])
    attempt = iter(range(1, group_size + 1))

    def make_env():
        conversation = Conversation(task, max_actions=row["max_actions"], max_exchanges=row["max_exchanges"],
                                    prompt_version=prompt_version)
        return MazeEnv(conversation, operator, renderer, recorder=Recorder(output / f"attempt-{next(attempt):02d}"),
                       max_tokens=max_tokens, context_tokens=context_tokens, timeout_seconds=timeout_seconds)

    envs = await MazeGroupBuilder(make_env, group_size).make_envs()
    results = await asyncio.gather(*(do_single_rollout(policy, env) for env in envs), return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        return None, errors  # infrastructure/budget interruptions never become reward-0 samples
    for env, trajectory in zip(envs, results):
        save_trajectory(trajectory, env.recorder.directory / "trajectory.json")
    return TrajectoryGroup(list(results), [0.] * group_size, [{} for _ in results]), []


async def update_policy(client, service, expert, guard, data, settings, name):
    """One importance-sampling backward pass and optimizer step, then sample from the new weights."""
    import tinker

    from panel_guide import train

    tokens = sum(d.model_input.length for d in data)
    guard.reserve_training(expert.config.model, tokens, settings["train_usd_per_million"], operation="forward_backward")
    backward = await (await client.forward_backward_async(data, loss_fn=settings["loss_fn"])).result_async()
    if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in backward.metrics.values()):
        raise ValueError("Nonfinite backward metric; no optimizer step.")
    step = await (await client.optim_step_async(tinker.AdamParams(
        learning_rate=settings["learning_rate"], beta1=.9, beta2=.95, eps=1e-8,
        grad_clip_norm=settings["grad_clip_norm"]))).result_async()
    checkpoint = await train.save_checkpoint(client, name, settings["checkpoint_ttl_seconds"])
    await train.attach_sampler(service, expert, checkpoint)
    return checkpoint, {**backward.metrics, **step.metrics}


def batch_record(number, rows, results, totals, guard):
    groups = [g for _, g, _ in results if g is not None]
    rewards = [r for g in groups for r in g.get_total_rewards()]
    invalid = sum("stop/protocol_failure" in t.transitions[-1].metrics
                  for g in groups for t in g.trajectories_G if t.transitions)
    return {"batch": number, "tasks": [r["task_id"] for r in rows], "groups": len(groups),
            "dropped_groups": len(rows) - len(groups), "mixed_groups": totals["mixed_groups"],
            "episodes": len(rewards), "successes": int(sum(rewards)), "protocol_failures": invalid,
            "expert_target_tokens": totals["expert_target_tokens"], "reserved_usd": round(guard.reserved_usd, 4)}


async def run(args):
    load_dotenv(dotenv_path=Path(".env"), override=False, interpolate=False)
    config = tomllib.loads(args.config.read_text())
    settings = config["training"]
    _, records = load_bank(args.bank)
    pool = training_pool(records, args.split, (args.min_route, args.max_route))
    plan = plan_summary(pool, settings, args.batches)
    if args.check:
        print(json.dumps({**plan, "tinker_api_key_present": bool(os.environ.get("TINKER_API_KEY")),
                          "note": "Local check only. No network requests."}, indent=2))
        return 0
    if args.budget_usd is None:
        raise ValueError("Training requires an explicit --budget-usd cap.")
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("Set TINKER_API_KEY in the project's .env or your local shell.")

    from panel_guide import train
    from panel_guide.tinker_backend import ModelConfig, SpendingGuard, create_agents
    from panel_guide.tinker_env import ExpertTokenPolicy

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or Path("outputs") / f"train-rl-{stamp}"
    events = Recorder(output)
    train.write_json(output / "run.json", {"config": config, "plan": plan, "bank": str(args.bank),
        "split": args.split, "routes": [args.min_route, args.max_route], "seed": args.seed,
        "resume": args.resume, "budget_usd": args.budget_usd})
    guard = SpendingGuard(args.budget_usd, record=events.event)
    sampler = TaskSampler(pool, group_size=settings["group_size"], seed=args.seed)
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    expert, operator, service = await create_agents(ModelConfig(**config["expert"]),
                                                    ModelConfig(**config["operator"]), guard)
    status = "interrupted"
    try:
        if args.resume:
            client = await service.create_training_client_from_state_async(args.resume)
        else:
            client = await service.create_lora_training_client_async(base_model=expert.config.model,
                                                                     rank=settings["lora_rank"])
        checkpoint = await train.save_checkpoint(client, "step-000", settings["checkpoint_ttl_seconds"])
        await train.attach_sampler(service, expert, checkpoint)
        events.event(type="checkpoint", step=0, checkpoint=checkpoint)
        policy = ExpertTokenPolicy(expert, timeout_seconds=config["timeout_seconds"])
        limit = asyncio.Semaphore(args.parallel_groups)

        async def group(row, folder):
            async with limit:
                return (row, *await collect_group(row, config["prompt_version"], policy, operator, expert.renderer,
                    folder / row["task_id"], group_size=settings["group_size"], max_tokens=expert.config.max_tokens,
                    context_tokens=expert.config.context_tokens, timeout_seconds=config["timeout_seconds"]))

        for number in range(1, args.batches + 1):
            rows = sampler.sample(settings["tasks_per_batch"])
            results = await asyncio.gather(*(group(row, output / f"batch-{number:02d}") for row in rows))
            for row, result, errors in results:
                if result is None:
                    events.event(type="group_dropped", batch=number, task_id=row["task_id"],
                                 errors=sorted({type(e).__name__ for e in errors}))
                else:
                    sampler.observe(row["task_id"], [int(r) for r in result.get_total_rewards()])
            data, _, totals = train.prepare_update([g for _, g, _ in results if g is not None],
                                                   settings["normalization"])
            record = batch_record(number, rows, results, totals, guard)
            if any(is_budget_stop(e) for _, _, errors in results for e in errors):
                status = "budget_stopped"
            elif record["episodes"] and record["protocol_failures"] / record["episodes"] > settings["stop_invalid_rate"]:
                status = "format_stopped"  # the failure mode of the token-summed loss at high LR
            elif data:
                checkpoint, metrics = await update_policy(client, service, expert, guard, data, settings,
                                                          f"step-{number:03d}")
                sampler.advance()
                record.update(checkpoint=checkpoint, update_metrics=metrics)
            events.event(type="batch", **record)
            print(f"batch {number}: {record['successes']}/{record['episodes']} solved, "
                  f"{record['mixed_groups']} mixed groups, ${guard.reserved_usd:.2f} reserved", flush=True)
            if status != "interrupted":
                break
        else:
            status = "complete"
    finally:
        train.write_json(output / "budget.json", guard.to_dict())
        train.write_json(output / "summary.json", {"status": status, "final_checkpoint": expert.model_path})
        await service.close("success" if status == "complete" else "interrupted")
    print(f"{status}; final sampler: {expert.model_path}\nRun directory: {output}")
    return 0 if status == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True, help="Maze bank from panel_guide.generate_mazes.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-route", type=int, default=5)
    parser.add_argument("--max-route", type=int, default=10)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--parallel-groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", help="Tinker training-state path to continue from.")
    parser.add_argument("--config", type=Path, default=Path("configs/maze.toml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--check", action="store_true", help="Show the plan and rough cost without network requests.")
    args = parser.parse_args()
    try:
        code = asyncio.run(run(args))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Setup error: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
