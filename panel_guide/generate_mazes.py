"""Create a frozen, solver-validated task bank without model calls."""

import argparse
from pathlib import Path

from panel_guide.engine import MazeGame
from panel_guide.manual import build_manual
from panel_guide.maze_bank import generate_bank, load_bank, write_bank
from panel_guide.render import crop_current_room, render_maze
from panel_guide.tasks import task_from_dict


def write_previews(directory: Path, records: list[dict]) -> None:
    """One training example per topology; private manuals stay separate from PNGs."""
    seen = set()
    preview_dir = directory / "previews"
    preview_dir.mkdir()
    for row in records:
        if row["split"] != "train" or row["topology_id"] in seen:
            continue
        seen.add(row["topology_id"])
        task = task_from_dict(row["task"])
        view = MazeGame(task).maze_view
        prefix = preview_dir / row["task_id"]
        with render_maze(view) as overview:
            overview.save(prefix.with_suffix(".png"))
            with crop_current_room(overview, view) as detail:
                detail.save(preview_dir / f"{row['task_id']}-detail.png")
        prefix.with_suffix(".md").write_text(build_manual(task) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topologies", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=(4, 2, 2))
    parser.add_argument("--mazes-per-topology", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5000)
    args = parser.parse_args()
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Use a new or empty dataset directory.")
        manifest, records = generate_bank(args.seed, tuple(args.topologies), args.mazes_per_topology, args.max_candidates)
        write_bank(args.output_dir, manifest, records)
        load_bank(args.output_dir)
        write_previews(args.output_dir, records)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Generation error: {exc}\n")
    print(f"Saved {len(manifest['mazes'])} mazes / {len(records)} episodes to {args.output_dir}")
    for split, summary in manifest["splits"].items():
        print(f"{split}: {summary['tasks']} tasks; {summary['difficulty']}")


if __name__ == "__main__":
    main()
