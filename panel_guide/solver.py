"""Small exact searches for auditing. Solver output is never an agent observation."""

from collections import deque
from dataclasses import dataclass

from panel_guide.engine import Action, local_actions, transition
from panel_guide.tasks import MazeTask, State


def successors(task: MazeTask, state: State) -> tuple[tuple[Action, State], ...]:
    if state.room == task.goal:
        return ()
    result = []
    for action in local_actions(task, state):
        after, _ = transition(task, state, action)
        if after != state:
            result.append((action, after))
    return tuple(result)


def solve(task: MazeTask, initial: State | None = None) -> tuple[Action, ...] | None:
    """Shortest physical route, or None when no solution exists."""
    start = task.initial if initial is None else initial
    task.room(start.room)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if state.room == task.goal:
            return path
        for action, after in successors(task, state):
            if after not in seen:
                seen.add(after)
                queue.append((after, (*path, action)))
    return None


def reachable_states(task: MazeTask) -> frozenset[State]:
    seen = {task.initial}
    queue = deque([task.initial])
    while queue:
        for _, after in successors(task, queue.popleft()):
            if after not in seen:
                seen.add(after)
                queue.append(after)
    return frozenset(seen)


def _can_solve_without_revisits(task: MazeTask) -> bool:
    start = (task.initial, frozenset((task.initial.room,)))
    queue = deque([start])
    seen = {start}
    while queue:
        state, visited = queue.popleft()
        if state.room == task.goal:
            return True
        for _, after in successors(task, state):
            if after.room != state.room and after.room in visited:
                continue
            entry = (after, visited | {after.room})
            if entry not in seen:
                seen.add(entry)
                queue.append(entry)
    return False


def _minimum_toggles(task: MazeTask) -> int | None:
    """0–1 search: moving costs zero; a toggle costs one."""
    costs = {task.initial: 0}
    queue = deque([task.initial])
    while queue:
        state = queue.popleft()
        if state.room == task.goal:
            return costs[state]
        for action, after in successors(task, state):
            cost = costs[state] + (action.kind == "toggle")
            if cost < costs.get(after, float("inf")):
                costs[after] = cost
                if action.kind == "toggle":
                    queue.append(after)
                else:
                    queue.appendleft(after)
    return None


@dataclass(frozen=True)
class TaskAnalysis:
    solution_length: int | None
    minimum_toggles: int | None
    requires_revisit: bool | None
    reachable_state_count: int
    dead_end_states: tuple[State, ...]
    within_action_budget: bool


def analyze_task(task: MazeTask, max_actions: int = 16) -> TaskAnalysis:
    if type(max_actions) is not int or max_actions <= 0:
        raise ValueError("max_actions must be a positive integer.")
    solution = solve(task)
    states = reachable_states(task)
    dead_ends = tuple(sorted(
        (state for state in states if solve(task, state) is None),
        key=lambda state: (state.room, state.switches),
    ))
    return TaskAnalysis(
        solution_length=len(solution) if solution is not None else None,
        minimum_toggles=_minimum_toggles(task),
        requires_revisit=not _can_solve_without_revisits(task) if solution is not None else None,
        reachable_state_count=len(states),
        dead_end_states=dead_ends,
        within_action_budget=solution is not None and len(solution) <= max_actions,
    )
