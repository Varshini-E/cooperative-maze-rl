"""Small development ladder on one map, not a procedural task distribution."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from panel_guide.tasks import MazeTask, State, example_task

REFERENCE_ROUTE = (
    ("move", "bars"), ("toggle", "striped circle"), ("move", "zigzag"),
    ("move", "ring"), ("move", "triangle"), ("move", "bars"),
    ("toggle", "solid circle"), ("move", "triangle"), ("move", "plus"),
)


@dataclass(frozen=True)
class Level:
    initial: State
    solution_length: int
    max_actions: int
    max_exchanges: int
    scripted_route: tuple[tuple[str, str], ...]


LEVELS = {
    "one-door": Level(State("Gallery", (True, True)), 1, 4, 8, REFERENCE_ROUTE[-1:]),
    "one-switch": Level(State("Tower", (True, False)), 3, 6, 16, REFERENCE_ROUTE[-3:]),
    "full": Level(State("Crescent", (False, False)), 9, 16, 32, REFERENCE_ROUTE),
}


def development_task(level: str = "full", layout: int = 0) -> MazeTask:
    """Rotate local action IDs reproducibly; room geometry and symbols stay fixed.

    Six layouts cover the joint cycle of two-door and three-door rooms. These
    are cosmetic variants of one topology, not independent structural holdouts.
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown task level: {level}")
    if type(layout) is not int or not 0 <= layout < 6:
        raise ValueError("layout must be an integer from 0 through 5.")
    task = example_task()
    rooms = []
    for room in task.rooms:
        offset = layout % len(room.doors)
        doors = room.doors[offset:] + room.doors[:offset]
        rooms.append(replace(room, doors=tuple(replace(door, id=f"door_{i}")
                                               for i, door in enumerate(doors, 1))))
    return replace(task, rooms=tuple(rooms), initial=LEVELS[level].initial)


def task_id(task: MazeTask) -> str:
    payload = json.dumps(asdict(task), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
