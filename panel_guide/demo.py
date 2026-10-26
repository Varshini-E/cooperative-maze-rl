"""Run the local maze fixture, including a blocked move, recovery, and timeout."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from panel_guide.engine import Action, MazeGame
from panel_guide.manual import build_manual
from panel_guide.render import DETAIL_RENDER_VERSION, crop_current_room, render_maze
from panel_guide.solver import analyze_task, solve
from panel_guide.tasks import example_task


def run_demo(output_dir: Path, max_actions: int = 16) -> dict:
    task = example_task()
    analysis = analyze_task(task, max_actions)
    solution = solve(task)
    assert solution is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manual.txt").write_text(build_manual(task) + "\n", encoding="utf-8")
    # Visit Garden too early, try its locked door, then return and recover.
    detour = (Action("move", "door_1"), Action("move", "door_2"), Action("move", "door_1"))
    scenarios = {
        "reference": solution,
        "blocked_and_recovery": (*detour, *solution),
        "budget_exhaustion": (Action("move", "door_1"),) + (Action("move", "door_2"),) * (max_actions - 1),
    }
    episodes = []
    for name, actions in scenarios.items():
        game = MazeGame(task, max_actions)
        episode_dir = output_dir / name
        episode_dir.mkdir(exist_ok=True)
        # These directories contain generated demo frames; clear stale frames on reruns.
        for stale in episode_dir.glob("frame-*.png"):
            stale.unlink()
        for stale in episode_dir.glob("detail-*.png"):
            stale.unlink()
        observations = []
        events = []

        def save_frame(index):
            relative = f"{name}/frame-{index:02d}.png"
            detail_path = f"{name}/detail-{index:02d}.png"
            with render_maze(game.maze_view) as image:
                image.save(output_dir / relative)
                with crop_current_room(image, game.maze_view) as detail:
                    detail.save(output_dir / detail_path)
            return {"image": relative, "detail_image": detail_path,
                    "view": asdict(game.view), "maze_view": asdict(game.maze_view)}

        observations.append(save_frame(0))
        for index, action in enumerate(actions, start=1):
            before = game.audit_state
            feedback = game.step(action)
            observation = save_frame(index)
            observation["feedback"] = asdict(feedback)
            observations.append(observation)
            events.append({
                "before": asdict(before),
                "feedback": asdict(feedback),
                "after": asdict(game.audit_state),
                "image": observation["image"],
                "detail_image": observation["detail_image"],
            })
            if game.done:
                break
        (episode_dir / "observations.json").write_text(
            json.dumps(observations, indent=2) + "\n", encoding="utf-8",
        )
        episodes.append({
            "name": name,
            "events": events,
            "status": game.status,
            "action_attempts": len(events),
            "total_reward": sum(event["feedback"]["reward"] for event in events),
        })
    audit = {
        "kind": "privileged_maze_demo_trace",
        "renderer": DETAIL_RENDER_VERSION,
        "observation_mode": "maze_detail",
        "task": asdict(task),
        "analysis": asdict(analysis),
        "solution": [asdict(action) for action in solution],
        "max_actions": max_actions,
        "episodes": episodes,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maze-overview-demo"))
    parser.add_argument("--max-actions", type=int, default=16)
    args = parser.parse_args()
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    trace = run_demo(args.output_dir, args.max_actions)
    analysis = trace["analysis"]
    print(f"Shortest solution: {analysis['solution_length']} actions; "
          f"minimum toggles: {analysis['minimum_toggles']}; "
          f"revisits required: {analysis['requires_revisit']}")
    for episode in trace["episodes"]:
        print(f"{episode['name']}: {episode['status']} | "
              f"actions={episode['action_attempts']} | reward={episode['total_reward']:g}")
    print(f"Saved manual, maze images, visible observations, and privileged audit to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
