"""Immutable maze specifications and the first hand-checked episode."""

from dataclasses import dataclass

SWITCHES = ("A", "B")
SYMBOLS = frozenset({
    "crescent", "anvil", "leaf", "sun", "star", "diamond",
    "bars", "ring", "zigzag", "plus", "triangle",
    "striped circle", "solid circle",
})


@dataclass(frozen=True)
class State:
    """Privileged physical state. Do not pass this to either agent."""

    room: str
    switches: tuple[bool, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "switches", tuple(self.switches))
        if not isinstance(self.room, str) or not self.room:
            raise ValueError("A state needs a room ID.")
        if len(self.switches) != 2 or any(type(v) is not bool for v in self.switches):
            raise ValueError("Switch values must be two booleans, in A/B order.")


@dataclass(frozen=True)
class Requirement:
    switch: str
    value: bool

    def __post_init__(self) -> None:
        if self.switch not in SWITCHES or type(self.value) is not bool:
            raise ValueError("A requirement needs switch A/B and a boolean value.")


@dataclass(frozen=True)
class Door:
    id: str
    symbol: str
    destination: str
    requirements: tuple[Requirement, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirements", tuple(sorted(self.requirements, key=lambda r: r.switch)))
        if len({r.switch for r in self.requirements}) != len(self.requirements):
            raise ValueError("A gate cannot specify the same switch twice.")


@dataclass(frozen=True)
class Control:
    id: str
    symbol: str
    switch: str

    def __post_init__(self) -> None:
        if self.switch not in SWITCHES:
            raise ValueError("A control must operate switch A or B.")


@dataclass(frozen=True)
class Room:
    id: str
    emblem: str
    doors: tuple[Door, ...]
    controls: tuple[Control, ...] = ()
    position: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doors", tuple(self.doors))
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "position", tuple(self.position))
        if len(self.position) != 2 or any(type(v) is not int or v < 0 for v in self.position):
            raise ValueError("Room position must be a pair of nonnegative grid coordinates.")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("A room needs a nonempty ID.")
        if self.emblem not in SYMBOLS:
            raise ValueError("Unsupported room emblem.")
        if not 1 <= len(self.doors) <= 3 or len(self.controls) > 2:
            raise ValueError("Rooms need one to three doors and at most two controls.")
        for prefix, items in (("door", self.doors), ("control", self.controls)):
            expected = tuple(f"{prefix}_{i + 1}" for i in range(len(items)))
            if tuple(item.id for item in items) != expected:
                raise ValueError("Local IDs must be consecutive, starting at 1.")
            if len({item.symbol for item in items}) != len(items):
                raise ValueError("Local symbols must be unambiguous within each object type.")
            if any(item.symbol not in SYMBOLS for item in items):
                raise ValueError("Unsupported local symbol.")
        if len({door.destination for door in self.doors}) != len(self.doors):
            raise ValueError("A room cannot have multiple doors to the same destination.")


@dataclass(frozen=True)
class MazeTask:
    """Static map and hidden initial state; validates structure, not solvability."""

    rooms: tuple[Room, ...]
    initial: State
    goal: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rooms", tuple(self.rooms))
        if len(self.rooms) != 6 or len({room.id for room in self.rooms}) != 6:
            raise ValueError("A maze needs six uniquely named rooms.")
        if len({room.emblem for room in self.rooms}) != 6:
            raise ValueError("Each room needs a distinct emblem.")
        if len({room.position for room in self.rooms}) != len(self.rooms):
            raise ValueError("Rooms must occupy distinct floor-plan cells.")
        room_ids = {room.id for room in self.rooms}
        if self.initial.room not in room_ids or self.goal not in room_ids:
            raise ValueError("Start and goal must name existing rooms.")
        if self.initial.room == self.goal:
            raise ValueError("An episode cannot start solved.")
        controls = [c.switch for room in self.rooms for c in room.controls]
        if sorted(controls) != list(SWITCHES):
            raise ValueError("Place exactly one control for each of switches A and B.")
        for room in self.rooms:
            for door in room.doors:
                if door.destination not in room_ids or door.destination == room.id:
                    raise ValueError("Doors must connect distinct existing rooms.")
                destination = self.room(door.destination)
                if sum(abs(a - b) for a, b in zip(room.position, destination.position)) != 1:
                    raise ValueError("Connected rooms must be adjacent on the floor plan.")
                reverse = next((d for d in self.room(door.destination).doors if d.destination == room.id), None)
                if reverse is None or reverse.requirements != door.requirements:
                    raise ValueError("Passages must have matching gates in both directions.")
        seen = {self.initial.room}
        pending = [self.initial.room]
        while pending:
            for door in self.room(pending.pop()).doors:
                if door.destination not in seen:
                    seen.add(door.destination)
                    pending.append(door.destination)
        if seen != room_ids:
            raise ValueError("The underlying room graph must be connected.")

    def room(self, room_id: str) -> Room:
        for room in self.rooms:
            if room.id == room_id:
                return room
        raise ValueError(f"Unknown room: {room_id}")


def task_from_dict(data: dict) -> MazeTask:
    """Reconstruct a saved task through the same structural validation as generation."""
    rooms = []
    for room in data["rooms"]:
        doors = tuple(Door(**{**door, "requirements": tuple(Requirement(**rule)
                      for rule in door["requirements"])}) for door in room["doors"])
        controls = tuple(Control(**control) for control in room["controls"])
        rooms.append(Room(**{**room, "doors": doors, "controls": controls}))
    return MazeTask(rooms=tuple(rooms), initial=State(**data["initial"]), goal=data["goal"])


def example_task() -> MazeTask:
    """README fixture: seven moves, two toggles, and two necessary revisits."""
    a = (Requirement("A", True),)
    ab = (*a, Requirement("B", True))
    return MazeTask(
        rooms=(
            Room("Crescent", "crescent", (
                Door("door_1", "ring", "Garden"),
                Door("door_2", "bars", "Workshop"),
            ), position=(1, 0)),
            Room("Workshop", "anvil", (Door("door_1", "zigzag", "Crescent"),), (
                Control("control_1", "striped circle", "A"),
            ), position=(0, 0)),
            Room("Garden", "leaf", (
                Door("door_1", "plus", "Crescent"),
                Door("door_2", "triangle", "Gallery", a),
            ), position=(1, 1)),
            Room("Gallery", "sun", (
                Door("door_1", "ring", "Garden", a),
                Door("door_2", "bars", "Tower"),
                Door("door_3", "plus", "Vault", ab),
            ), position=(2, 1)),
            Room("Tower", "star", (Door("door_1", "triangle", "Gallery"),), (
                Control("control_1", "solid circle", "B"),
            ), position=(2, 0)),
            Room("Vault", "diamond", (Door("door_1", "zigzag", "Gallery", ab),), position=(3, 1)),
        ),
        initial=State("Crescent", (False, False)),
        goal="Vault",
    )
