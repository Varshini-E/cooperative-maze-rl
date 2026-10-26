"""Run one recorded conversation. Scripted mode is free and is the default."""

import argparse
import asyncio
import json
import os
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from panel_guide.conversation import Conversation, Recorder
from panel_guide.curriculum import LEVELS, development_task, task_id
from panel_guide.render import render_version
from panel_guide.reporting import dependency_versions
from panel_guide.scripted import COMMAND_CONTRACT, DiagnosticOperator, ScriptedExpert, ScriptedOperator
from panel_guide.solver import analyze_task
from panel_guide.tinker_backend import ModelConfig, SpendingGuard, create_agents


async def run(args: argparse.Namespace) -> int:
    # Parse only the project's .env; never execute it or replace exported values.
    load_dotenv(dotenv_path=Path(".env"), override=False, interpolate=False)
    config = tomllib.loads(args.config.read_text())
    if getattr(args, "operator_history", None):
        config["operator_history"] = args.operator_history
    expert_config = ModelConfig(**config["expert"])
    operator_config = ModelConfig(**config["operator"])
    if args.check:
        print(json.dumps({"dependencies": dependency_versions(),
                          "tinker_api_key_present": bool(os.environ.get("TINKER_API_KEY")),
                          "expert": expert_config.model, "operator": operator_config.model,
                          "note": "Local checks only. No account query or paid inference."}, indent=2))
        return 0
    task = development_task(args.task, args.layout)
    level = LEVELS[args.task]
    diagnostic = args.backend == "tinker" and args.partner == "scripted"
    runner = getattr(args, "runner", None) or ("cookbook" if args.backend == "tinker" and args.partner != "guided" else "conversation")
    if runner == "cookbook" and (args.backend != "tinker" or args.partner == "guided"):
        raise ValueError("The Cookbook runner samples a real Expert; use conversation for scripted Expert diagnostics.")
    environment_version = None
    if runner == "cookbook":
        from panel_guide.tinker_env import ENV_VERSION, run_cookbook_episode
        environment_version = ENV_VERSION
    max_actions = min(config["max_actions"], level.max_actions)
    analysis = analyze_task(task, max_actions)
    if not analysis.within_action_budget or analysis.dead_end_states:
        raise ValueError("Task must be solvable within budget and have no reachable dead ends.")
    conversation = Conversation(task, max_actions=max_actions,
                                max_exchanges=min(config["max_exchanges"], level.max_exchanges),
                                prompt_version=config.get("prompt_version", "v1"),
                                observation_mode=config.get("observation_mode", "maze"),
                                operator_history=config.get("operator_history", "full"),
                                expert_instructions=COMMAND_CONTRACT if diagnostic else "")
    budget = None
    service = None
    if args.backend == "tinker":
        if args.budget_usd is None:
            raise ValueError("Tinker requires an explicit --budget-usd allocation for this episode.")
        budget = SpendingGuard(args.budget_usd)
        if not os.environ.get("TINKER_API_KEY"):
            raise ValueError("Set TINKER_API_KEY in the project's .env or your local shell.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_dir or Path("outputs") / f"conversation-{args.backend}-{stamp}"
    recorder = Recorder(output)
    recorder.audit(type="task", task=asdict(task), analysis=asdict(analysis))
    (output / "run.json").write_text(json.dumps({"backend": args.backend, "runner": runner, "config": config,
        "environment_version": environment_version,
        "task": args.task, "layout": args.layout, "task_id": task_id(task),
        "observation_mode": conversation.observation_mode, "renderer": render_version(conversation.observation_mode),
        "partner": args.partner if args.backend == "tinker" else "scripted",
        "dependencies": dependency_versions(), "budget_usd": args.budget_usd}, indent=2) + "\n")
    if budget:
        budget.record = recorder.event
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        try:
            expert, operator, service = await create_agents(expert_config, operator_config, budget,
                                                           capture_dir=output / "provider-requests")
            if diagnostic:
                operator = DiagnosticOperator()
            elif args.partner == "guided":
                expert = ScriptedExpert(level.scripted_route)
        except Exception as exc:
            recorder.event(type="setup_error", error_class=type(exc).__name__)
            raise
    else:
        expert, operator = ScriptedExpert(level.scripted_route), ScriptedOperator()
    result = None
    try:
        if runner == "cookbook":
            result = await run_cookbook_episode(conversation, expert, operator, recorder,
                                                timeout_seconds=config["timeout_seconds"])
        else:
            result = await conversation.run(expert, operator, recorder, timeout_seconds=config["timeout_seconds"])
    finally:
        if budget:
            (output / "budget.json").write_text(json.dumps(budget.to_dict(), indent=2) + "\n")
        if service:
            await service.close("interrupted" if result is None or result.status == "interrupted" else "success")
    print(f"{result.status}: {result.reason} | exchanges={result.exchanges} | "
          f"actions={result.actions} | reward={result.reward:g}")
    print(f"Trace: {output / 'trace.jsonl'}")
    return 2 if result.status == "interrupted" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("scripted", "tinker"), default="scripted")
    parser.add_argument("--runner", choices=("cookbook", "conversation"),
                        help="Real Expert defaults to Cookbook Env; scripted Expert uses the conversation runner.")
    parser.add_argument("--task", choices=tuple(LEVELS), default="full")
    parser.add_argument("--layout", type=int, choices=range(6), default=0)
    parser.add_argument("--partner", choices=("visual", "scripted", "guided"), default="visual",
                        help="Tinker: real pair, real Expert with scripted observations, or scripted guide with real VLM.")
    parser.add_argument("--operator-history", choices=("full", "latest", "recent2"),
                        help="Override how much shared text reaches the Operator; Expert retains full history.")
    parser.add_argument("--config", type=Path, default=Path("configs/maze.toml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--check", action="store_true", help="Check local setup without any network requests.")
    args = parser.parse_args()
    try:
        code = asyncio.run(run(args))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Setup error: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
