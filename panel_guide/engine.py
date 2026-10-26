"""Physical transitions, sparse rewards, and observations with local state visibility."""

from dataclasses import dataclass
from typing import Literal

from panel_guide.tasks import Door, MazeTask, State, SWITCHES

Status = Literal["active", "success", "failure"]
Outcome = Literal["moved", "toggled", "blocked", "invalid_target"]


@dataclass(frozen=True)
class Action:
    kind: Literal["move", "toggle"]
    target: str

    def __post_init__(self) -> None:
        if self.kind not in ("move", "toggle") or not isinstance(self.target, str) or not self.target:
            raise ValueError("An action needs move/toggle and a nonempty local target ID.")


@dataclass(frozen=True)
class VisibleDoor:
    id: str
    symbol: str
    is_open: bool


@dataclass(frozen=True)
class VisibleControl:
    id: str
    symbol: str
    is_on: bool


@dataclass(frozen=True)
class RoomView:
    """Current-room fields used by scripted diagnostics and the legacy renderer."""

    emblem: str
    doors: tuple[VisibleDoor, ...]
    controls: tuple[VisibleControl, ...]


@dataclass(frozen=True)
class MapDoor:
    id: str | None
    symbol: str
    neighbor: tuple[int, int]
    is_open: bool | None


@dataclass(frozen=True)
class MapControl:
    id: str | None
    symbol: str
    is_on: bool | None


@dataclass(frozen=True)
class MapRoom:
    position: tuple[int, int]
    emblem: str
    current: bool
    doors: tuple[MapDoor, ...]
    controls: tuple[MapControl, ...]


@dataclass(frozen=True)
class MazeView:
    """Public geometry and symbols. Remote states/targets are absent, not cached.

    No manual room names, goal, switch A/B identities, gate rules, or solution.
    """

    rooms: tuple[MapRoom, ...]


@dataclass(frozen=True)
class StepResult:
    """Public execution feedback, with no room identity or switch values."""

    action: Action
    outcome: Outcome
    remaining_actions: int
    status: Status
    reward: float
    termination_reason: Literal["goal_reached", "action_limit"] | None

    @property
    def done(self) -> bool:
        return self.status != "active"


def door_is_open(door: Door, state: State) -> bool:
    return all(state.switches[SWITCHES.index(r.switch)] == r.value for r in door.requirements)


def observe(task: MazeTask, state: State) -> RoomView:
    room = task.room(state.room)
    return RoomView(
        emblem=room.emblem,
        doors=tuple(VisibleDoor(d.id, d.symbol, door_is_open(d, state)) for d in room.doors),
        controls=tuple(VisibleControl(c.id, c.symbol, state.switches[SWITCHES.index(c.switch)]) for c in room.controls),
    )


def observe_maze(task: MazeTask, state: State) -> MazeView:
    rooms = []
    for room in task.rooms:
        current = room.id == state.room
        rooms.append(MapRoom(
            position=room.position, emblem=room.emblem, current=current,
            doors=tuple(MapDoor(
                d.id if current else None, d.symbol, task.room(d.destination).position,
                door_is_open(d, state) if current else None,
            ) for d in room.doors),
            controls=tuple(MapControl(
                c.id if current else None, c.symbol,
                state.switches[SWITCHES.index(c.switch)] if current else None,
            ) for c in room.controls),
        ))
    return MazeView(tuple(rooms))


def local_actions(task: MazeTask, state: State) -> tuple[Action, ...]:
    room = task.room(state.room)
    return (
        *(Action("move", door.id) for door in room.doors),
        *(Action("toggle", control.id) for control in room.controls),
    )


def transition(task: MazeTask, state: State, action: Action) -> tuple[State, Outcome]:
    """Pure physical transition. Counters, termination, and reward live in MazeGame."""
    if not isinstance(action, Action):
        raise TypeError("Pass a parsed Action to the engine.")
    room = task.room(state.room)
    if action.kind == "move":
        door = next((d for d in room.doors if d.id == action.target), None)
        if door is None:
            return state, "invalid_target"
        if not door_is_open(door, state):
            return state, "blocked"
        return State(door.destination, state.switches), "moved"
    control = next((c for c in room.controls if c.id == action.target), None)
    if control is None:
        return state, "invalid_target"
    values = list(state.switches)
    index = SWITCHES.index(control.switch)
    values[index] = not values[index]
    return State(state.room, tuple(values)), "toggled"


class MazeGame:
    """One episode with a physical-action budget. Conversation owns message limits."""

    def __init__(self, task: MazeTask, max_actions: int = 16) -> None:
        if type(max_actions) is not int or max_actions <= 0:
            raise ValueError("max_actions must be a positive integer.")
        self._task = task
        self._state = task.initial
        self._max_actions = max_actions
        self._attempts = 0
        self._status: Status = "active"

    @property
    def audit_state(self) -> State:
        """Privileged state for debugging and oracle code, never model input."""
        return self._state

    @property
    def view(self) -> RoomView:
        return observe(self._task, self._state)

    @property
    def maze_view(self) -> MazeView:
        return observe_maze(self._task, self._state)

    @property
    def status(self) -> Status:
        return self._status

    @property
    def done(self) -> bool:
        return self._status != "active"

    @property
    def remaining_actions(self) -> int:
        return self._max_actions - self._attempts

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("The episode has ended; create a new MazeGame to restart.")
        state, outcome = transition(self._task, self._state, action)
        self._state = state
        self._attempts += 1
        reward = 0.0
        reason = None
        if state.room == self._task.goal:
            self._status = "success"
            reward = 1.0
            reason = "goal_reached"
        elif self.remaining_actions == 0:
            self._status = "failure"
            reason = "action_limit"
        return StepResult(action, outcome, self.remaining_actions, self.status, reward, reason)
