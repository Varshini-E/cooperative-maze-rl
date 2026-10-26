"""Seeded six-room puzzles, structural split isolation, and frozen task banks.

All solver metadata is audit-only. No models are called here. Sampling is a
bounded constructive search with rejection; it is not uniform over all mazes.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import combinations, permutations, product
from pathlib import Path

from panel_guide.curriculum import task_id
from panel_guide.solver import analyze_task
from panel_guide.tasks import Control, Door, MazeTask, Requirement, Room, State, task_from_dict

GENERATOR_VERSION = "six-room-bank-v1"
SPLITS = ("train", "validation_starts", "validation_mazes", "test_starts", "test_mazes")
DIFFICULTIES = {"easy": (1, 3), "medium": (4, 6), "hard": (7, 10)}
EMBLEMS = ("crescent", "anvil", "leaf", "sun", "star", "diamond")
DOOR_SYMBOLS = ("bars", "ring", "zigzag", "plus", "triangle")
PAIRS = tuple(combinations(range(6), 2))
ORDERS = tuple(permutations(range(6)))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


@lru_cache(maxsize=32768)
def topology_key(edges: tuple[tuple[int, int], ...]) -> str:
    """Exact graph-isomorphism canonicalization; six nodes need only 720 orders."""
    adjacent = [[0] * 6 for _ in range(6)]
    for a, b in edges:
        adjacent[a][b] = adjacent[b][a] = 1
    return min("".join(str(adjacent[order[a]][order[b]]) for a, b in PAIRS) for order in ORDERS)


def topology_id(task: MazeTask) -> str:
    indices = {room.id: i for i, room in enumerate(task.rooms)}
    edges = tuple(sorted({tuple(sorted((indices[room.id], indices[door.destination])))
                          for room in task.rooms for door in room.doors}))
    return digest(topology_key(edges))


def maze_id(task: MazeTask) -> str:
    """Ignore starts, appearance, positions, and room names; retain goal and rules."""
    indices = {room.id: i for i, room in enumerate(task.rooms)}
    nodes = [4 * (room.id == task.goal) + sum(1 if c.switch == "A" else 2 for c in room.controls)
             for room in task.rooms]
    adjacent = [[0] * 6 for _ in range(6)]
    for i, room in enumerate(task.rooms):
        for door in room.doors:
            code = 1 + sum({("A", False): 1, ("A", True): 2, ("B", False): 4, ("B", True): 8}[
                rule.switch, rule.value] for rule in door.requirements)
            adjacent[i][indices[door.destination]] = code
    return digest(min(tuple(nodes[i] for i in order)
                      + tuple(adjacent[order[a]][order[b]] for a, b in PAIRS) for order in ORDERS))


def difficulty(solution_length: int) -> str:
    for name, (low, high) in DIFFICULTIES.items():
        if low <= solution_length <= high:
            return name
    raise ValueError("Solution length is outside the frozen 1–10 action bands.")


def sample_floorplan(seed: int) -> tuple[tuple, tuple]:
    """Grow a spanning tree in a 4x3 grid, then optionally add adjacent passages."""
    rng = random.Random(seed)
    cells = [(x, y) for x in range(4) for y in range(3)]
    positions = [rng.choice(cells)]
    edges, degrees = [], [0]
    while len(positions) < 6:
        frontier = [(i, cell) for i, pos in enumerate(positions) if degrees[i] < 3
                    for cell in cells if cell not in positions and sum(abs(a - b) for a, b in zip(pos, cell)) == 1]
        parent, cell = rng.choice(frontier)
        edges.append((parent, len(positions)))
        positions.append(cell)
        degrees[parent] += 1
        degrees.append(1)
    extras = [pair for pair in PAIRS if pair not in edges and
              sum(abs(a - b) for a, b in zip(positions[pair[0]], positions[pair[1]])) == 1]
    rng.shuffle(extras)
    for a, b in extras:
        if degrees[a] < 3 and degrees[b] < 3 and rng.random() < .5:
            edges.append((a, b))
            degrees[a] += 1
            degrees[b] += 1
    return tuple(positions), tuple(sorted(edges))


def decorate(positions: tuple, edges: tuple, rule_seed: int, appearance_seed: int) -> MazeTask:
    rng, appearance = random.Random(rule_seed), random.Random(appearance_seed)
    goal = rng.randrange(6)
    controls = rng.sample([i for i in range(6) if i != goal], 2)
    gated = rng.sample(list(edges), rng.randint(2, min(3, len(edges))))
    rules = {edge: () for edge in edges}
    for index, edge in enumerate(gated):
        switches = ("A",) if index == 0 else ("B",) if index == 1 else rng.choice((("A",), ("B",), ("A", "B")))
        rules[edge] = tuple(Requirement(switch, bool(rng.getrandbits(1))) for switch in switches)
    emblems = appearance.sample(EMBLEMS, 6)
    control_symbols = appearance.sample(("striped circle", "solid circle"), 2)
    rooms = []
    for i, position in enumerate(positions):
        neighbors = [b if a == i else a for a, b in edges if i in (a, b)]
        appearance.shuffle(neighbors)
        symbols = appearance.sample(DOOR_SYMBOLS, len(neighbors))
        doors = tuple(Door(f"door_{j + 1}", symbols[j], f"Room {neighbor + 1}", rules[tuple(sorted((i, neighbor)))])
                      for j, neighbor in enumerate(neighbors))
        local_controls = tuple(Control("control_1", control_symbols[j], switch)
                               for j, switch in enumerate(("A", "B")) if controls[j] == i)
        rooms.append(Room(f"Room {i + 1}", emblems[i], doors, local_controls, position))
    start = next(room.id for room in rooms if room.id != f"Room {goal + 1}")
    return MazeTask(tuple(rooms), State(start, (False, False)), f"Room {goal + 1}")


def analyze_starts(task: MazeTask) -> tuple[list[tuple[MazeTask, dict]], str | None]:
    """Keep all 20 non-goal reset states, or reject the whole candidate maze."""
    episodes = []
    for room in task.rooms:
        if room.id == task.goal:
            continue
        for switches in product((False, True), repeat=2):
            episode = replace(task, initial=State(room.id, switches))
            analysis = analyze_task(episode)
            if analysis.solution_length is None or analysis.dead_end_states:
                return [], "unreachable_or_trap"
            if not 1 <= analysis.solution_length <= 10:
                return [], "outside_difficulty_range"
            episodes.append((episode, {key: getattr(analysis, key) for key in
                ("solution_length", "minimum_toggles", "requires_revisit")}))
    if max(a["minimum_toggles"] for _, a in episodes) < 2 or not any(a["requires_revisit"] for _, a in episodes):
        return [], "insufficient_switch_dependency"
    if {difficulty(a["solution_length"]) for _, a in episodes} != set(DIFFICULTIES):
        return [], "missing_difficulty_band"
    return episodes, None


def generate_bank(seed: int = 0, topology_counts: tuple[int, int, int] = (4, 2, 2),
                  mazes_per_topology: int = 3, max_candidates: int = 5000) -> tuple[dict, list[dict]]:
    if (type(seed) is not int or len(topology_counts) != 3
            or any(type(n) is not int or n <= 0 for n in (*topology_counts, mazes_per_topology, max_candidates))):
        raise ValueError("Use an integer seed and positive integer counts for all three topology partitions.")
    rng = random.Random(seed)
    topologies, rejected = {}, Counter()
    for _ in range(max_candidates):
        floor_seed = rng.getrandbits(63)
        positions, edges = sample_floorplan(floor_seed)
        identity = digest(topology_key(edges))
        if identity in topologies:
            rejected["duplicate_topology"] += 1
            continue
        topologies[identity] = (floor_seed, positions, edges)
        if len(topologies) == sum(topology_counts):
            break
    if len(topologies) < sum(topology_counts):
        raise ValueError(f"Found only {len(topologies)} distinct topologies; lower the requested counts or increase max_candidates.")
    families = list(topologies)
    rng.shuffle(families)
    partitions = dict(zip(families, ["train"] * topology_counts[0]
                          + ["validation_mazes"] * topology_counts[1] + ["test_mazes"] * topology_counts[2]))
    records, mazes, accepted_ids = [], [], set()
    for family in families:
        floor_seed, positions, edges = topologies[family]
        accepted = 0
        for _ in range(max_candidates):
            rule_seed, appearance_seed = rng.getrandbits(63), rng.getrandbits(63)
            task = decorate(positions, edges, rule_seed, appearance_seed)
            identity = maze_id(task)
            if identity in accepted_ids:
                rejected["duplicate_maze"] += 1
                continue
            episodes, reason = analyze_starts(task)
            if reason:
                rejected[reason] += 1
                continue
            rooms = [room.id for room in task.rooms if room.id != task.goal]
            rng.shuffle(rooms)
            partition = partitions[family]
            starts = (dict(zip(rooms, ["train"] * 3 + ["validation_starts", "test_starts"]))
                      if partition == "train" else dict.fromkeys(rooms, partition))
            mazes.append({"maze_id": identity, "topology_id": family, "partition": partition,
                          "floor_seed": floor_seed, "rule_seed": rule_seed, "appearance_seed": appearance_seed,
                          "start_splits": starts})
            for episode, analysis in episodes:
                records.append({"task_id": task_id(episode), "maze_id": identity, "topology_id": family,
                    "split": starts[episode.initial.room], "difficulty": difficulty(analysis["solution_length"]),
                    **analysis, "max_actions": 16, "max_exchanges": 32, "task": asdict(episode)})
            accepted_ids.add(identity)
            accepted += 1
            if accepted == mazes_per_topology:
                break
        if accepted != mazes_per_topology:
            raise ValueError(f"Accepted only {accepted} mazes for topology {family}; exhausted max_candidates.")
    manifest = {"generator_version": GENERATOR_VERSION, "seed": seed,
                "topology_counts": list(topology_counts), "mazes_per_topology": mazes_per_topology,
                "max_candidates": max_candidates, "difficulty_bands": DIFFICULTIES,
                "mazes": mazes, "rejections": dict(rejected),
                "splits": {split: {"tasks": sum(r["split"] == split for r in records),
                    "difficulty": dict(Counter(r["difficulty"] for r in records if r["split"] == split))}
                    for split in SPLITS},
                "note": "Seeded rejection sampling; all 20 non-goal resets valid per maze. "
                        "Topology partitions ignore names, symbols, positions, rules and goals. "
                        "New-start partitions reserve entire start rooms, not rooms visited during play. "
                        "Difficulty histograms are reported, not forced equal. No model calls or training."}
    return manifest, records


def write_bank(directory: Path, manifest: dict, records: list[dict]) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new or empty dataset directory; frozen banks are never overwritten.")
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    manifest = {**manifest, "tasks_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(Path(__file__).parent.glob("*.py"))}}
    (directory / "tasks.jsonl").write_text(payload)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_bank(directory: Path) -> tuple[dict, list[dict]]:
    """Validate serialized identities and both split boundaries before selection."""
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = (directory / "tasks.jsonl").read_bytes()
    if manifest["generator_version"] != GENERATOR_VERSION or hashlib.sha256(payload).hexdigest() != manifest["tasks_sha256"]:
        raise ValueError("Dataset version or task checksum mismatch.")
    records = [json.loads(line) for line in payload.splitlines()]
    maze_specs = {row["maze_id"]: row for row in manifest["mazes"]}
    if len(maze_specs) != len(manifest["mazes"]):
        raise ValueError("Duplicate maze identities.")
    family_partitions, seen_tasks, maze_tasks, reset_sets = {}, set(), {}, defaultdict(set)
    for row in records:
        task = task_from_dict(row["task"])
        family, identity = topology_id(task), maze_id(task)
        spec = maze_specs.get(identity)
        if (row["task_id"] != task_id(task) or row["task_id"] in seen_tasks or row["topology_id"] != family
                or row["maze_id"] != identity or spec is None or spec["topology_id"] != family):
            raise ValueError("Task identity mismatch or duplicate episode.")
        split = row["split"]
        if split not in SPLITS or spec["start_splits"].get(task.initial.room) != split:
            raise ValueError("Starting-room split mismatch.")
        partition = "train" if split in ("train", "validation_starts", "test_starts") else split
        if spec["partition"] != partition or family_partitions.setdefault(family, partition) != partition:
            raise ValueError("Equivalent topologies cross structural split boundaries.")
        static = {"rooms": row["task"]["rooms"], "goal": task.goal}
        if maze_tasks.setdefault(identity, static) != static:
            raise ValueError("A maze changes appearance or rules between starting-state splits.")
        reset = (task.initial.room, task.initial.switches)
        if reset in reset_sets[identity]:
            raise ValueError("A reset state appears more than once.")
        reset_sets[identity].add(reset)
        analysis = analyze_task(task, row["max_actions"])
        if (analysis.dead_end_states or not analysis.within_action_budget or row["max_exchanges"] != 32
                or row["max_actions"] != 16 or row["difficulty"] != difficulty(analysis.solution_length)
                or any(row[key] != getattr(analysis, key) for key in
                       ("solution_length", "minimum_toggles", "requires_revisit"))):
            raise ValueError("Invalid difficulty, solver metadata, or budgets.")
        seen_tasks.add(row["task_id"])
    if set(reset_sets) != set(maze_specs) or any(len(resets) != 20 for resets in reset_sets.values()):
        raise ValueError("Every maze must contain all 20 distinct non-goal reset states.")
    for identity, spec in maze_specs.items():
        expected = {"train": 3, "validation_starts": 1, "test_starts": 1} if spec["partition"] == "train" else {spec["partition"]: 5}
        if dict(Counter(spec["start_splits"].values())) != expected:
            raise ValueError("Starting rooms must follow the frozen 3/1/1 split.")
    if ([sum(p == partition for p in family_partitions.values())
         for partition in ("train", "validation_mazes", "test_mazes")] != manifest["topology_counts"]
            or any(count != manifest["mazes_per_topology"]
                   for count in Counter(spec["topology_id"] for spec in maze_specs.values()).values())):
        raise ValueError("Topology or maze counts do not match the manifest.")
    for split in SPLITS:
        rows = [row for row in records if row["split"] == split]
        if manifest["splits"][split] != {"tasks": len(rows), "difficulty": dict(Counter(row["difficulty"] for row in rows))}:
            raise ValueError("Split counts do not match the task bank.")
    return manifest, records


def select_tasks(records: list[dict], split: str = "train", limit: int = 12,
                 seed: int = 0, difficulties: tuple[str, ...] = tuple(DIFFICULTIES)) -> list[dict]:
    """Round-robin difficulty and maze IDs; sample without replacement within a batch."""
    if split not in SPLITS or type(limit) is not int or limit <= 0 or not difficulties or len(set(difficulties)) != len(difficulties):
        raise ValueError("Choose a valid split, a positive task limit, and distinct difficulty bands.")
    if any(name not in DIFFICULTIES for name in difficulties):
        raise ValueError("Unknown difficulty band.")
    rng = random.Random(seed)
    bands = {}
    for band in difficulties:
        by_maze = defaultdict(list)
        for row in records:
            if row["split"] == split and row["difficulty"] == band:
                by_maze[row["maze_id"]].append(row)
        if not by_maze:
            raise ValueError(f"No {band} tasks in {split}; choose available bands explicitly.")
        maze_ids = sorted(by_maze)
        rng.shuffle(maze_ids)
        for rows in by_maze.values():
            rng.shuffle(rows)
        ordered = []
        while any(by_maze.values()):
            for identity in maze_ids:
                if by_maze[identity]:
                    ordered.append(by_maze[identity].pop())
        bands[band] = iter(ordered)
    selected = []
    while bands and len(selected) < limit:
        for band in list(bands):
            row = next(bands[band], None)
            if row is None:
                del bands[band]
            else:
                selected.append(row)
                if len(selected) == limit:
                    break
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} tasks but only {len(selected)} are available in the selected split/bands.")
    return selected
