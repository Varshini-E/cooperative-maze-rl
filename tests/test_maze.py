from dataclasses import FrozenInstanceError, asdict, replace
from itertools import product
import unittest

from panel_guide.engine import Action, MazeGame, door_is_open, observe, transition
from panel_guide.manual import build_manual
from panel_guide.solver import analyze_task, solve
from panel_guide.tasks import Door, Requirement, Room, State, example_task


# Hand-written, independent of the solver and its transition search.
REFERENCE = (
    Action("move", "door_2"), Action("toggle", "control_1"),
    Action("move", "door_1"), Action("move", "door_1"),
    Action("move", "door_2"), Action("move", "door_2"),
    Action("toggle", "control_1"), Action("move", "door_1"),
    Action("move", "door_3"),
)
EXPECTED = (
    State("Workshop", (False, False)), State("Workshop", (True, False)),
    State("Crescent", (True, False)), State("Garden", (True, False)),
    State("Gallery", (True, False)), State("Tower", (True, False)),
    State("Tower", (True, True)), State("Gallery", (True, True)),
    State("Vault", (True, True)),
)


class SpecificationTests(unittest.TestCase):
    def test_nested_inputs_are_copied_and_frozen(self):
        task = example_task()
        values = [False, False]
        state = State("Crescent", values)
        values[0] = True
        rooms = list(task.rooms)
        copy = replace(task, rooms=rooms, initial=state)
        rooms.pop()
        self.assertEqual(len(copy.rooms), 6)
        self.assertEqual(copy.initial.switches, (False, False))
        with self.assertRaises(FrozenInstanceError):
            copy.goal = "Tower"

    def test_invalid_structure_is_rejected(self):
        task = example_task()
        invalid = (
            lambda: replace(task, goal="missing"),
            lambda: replace(task, goal=task.initial.room),
            lambda: replace(task, rooms=task.rooms[:-1]),
            lambda: replace(task, rooms=(task.rooms[0], *task.rooms[:-1])),
            lambda: replace(task, rooms=tuple(replace(r, controls=()) if r.id == "Tower" else r for r in task.rooms)),
            lambda: State("Crescent", (0, 1)),
            lambda: Requirement("C", True),
            lambda: Requirement("A", 1),
            lambda: Door("door_1", "bars", "Garden", (Requirement("A", True), Requirement("A", False))),
            lambda: Room("x", "sun", (Door("door_2", "ring", "y"),)),
            lambda: Room("x", "sun", (Door("door_1", "ring", "y"), Door("door_2", "ring", "z"))),
        )
        for make in invalid:
            with self.subTest(make=make), self.assertRaises(ValueError):
                make()

    def test_asymmetric_gates_are_rejected(self):
        task = example_task()
        garden = task.room("Garden")
        changed = replace(garden, doors=(garden.doors[0], replace(garden.doors[1], requirements=())))
        with self.assertRaises(ValueError):
            replace(task, rooms=tuple(changed if r.id == "Garden" else r for r in task.rooms))

    def test_manual_hides_initial_state_and_physical_slot_order(self):
        task = example_task()
        initial_changed = replace(task, initial=State("Tower", (True, True)))
        room = task.room("Crescent")
        swapped = replace(room, doors=(replace(room.doors[1], id="door_1"), replace(room.doors[0], id="door_2")))
        slots_changed = replace(task, rooms=tuple(swapped if r.id == room.id else r for r in task.rooms))
        self.assertEqual(build_manual(task), build_manual(initial_changed))
        self.assertEqual(build_manual(task), build_manual(slots_changed))
        self.assertNotEqual(observe(task, task.initial), observe(slots_changed, slots_changed.initial))
        self.assertNotIn("door_1", build_manual(task))
        self.assertNotIn("control_1", build_manual(task))
        self.assertIn("plus door -> Vault; requires A on and B on.", build_manual(task))

    def test_visible_view_contains_only_local_visual_fields(self):
        view = asdict(observe(example_task(), State("Workshop", (False, True))))
        self.assertEqual(view, {
            "emblem": "anvil",
            "doors": ({"id": "door_1", "symbol": "zigzag", "is_open": True},),
            "controls": ({"id": "control_1", "symbol": "striped circle", "is_on": False},),
        })


class EngineTests(unittest.TestCase):
    def test_reference_states_sparse_reward_and_terminal_boundary(self):
        game = MazeGame(example_task(), max_actions=9)
        for index, (action, expected) in enumerate(zip(REFERENCE, EXPECTED, strict=True)):
            result = game.step(action)
            self.assertEqual(game.audit_state, expected)
            self.assertEqual(result.reward, 1.0 if index == 8 else 0.0)
            self.assertEqual(result.done, index == 8)
        self.assertEqual(result.remaining_actions, 0)
        self.assertEqual(result.termination_reason, "goal_reached")
        self.assertEqual(result.status, "success")
        with self.assertRaises(RuntimeError):
            game.step(Action("move", "door_1"))
        self.assertEqual(game.audit_state, EXPECTED[-1])

    def test_blocked_move_costs_an_attempt_and_allows_recovery(self):
        game = MazeGame(example_task())
        game.step(Action("move", "door_1"))
        blocked = game.step(Action("move", "door_2"))
        self.assertEqual(blocked.outcome, "blocked")
        self.assertEqual(blocked.remaining_actions, 14)
        self.assertEqual(game.audit_state, State("Garden", (False, False)))
        self.assertFalse(blocked.done)
        game.step(Action("move", "door_1"))
        for action in REFERENCE:
            result = game.step(action)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.remaining_actions, 4)

    def test_switches_persist_and_can_be_turned_off(self):
        task = example_task()
        state = State("Workshop", (False, True))
        state, _ = transition(task, state, Action("toggle", "control_1"))
        self.assertEqual(state.switches, (True, True))
        state, _ = transition(task, state, Action("move", "door_1"))
        state, _ = transition(task, state, Action("move", "door_2"))
        state, _ = transition(task, state, Action("toggle", "control_1"))
        self.assertEqual(state, State("Workshop", (False, True)))

    def test_gate_truth_tables_include_off_requirements(self):
        doors = (
            Door("door_1", "ring", "x"),
            Door("door_1", "ring", "x", (Requirement("A", True),)),
            Door("door_1", "ring", "x", (Requirement("A", True), Requirement("B", True))),
            Door("door_1", "ring", "x", (Requirement("A", True), Requirement("B", False))),
        )
        for a, b in product((False, True), repeat=2):
            with self.subTest(a=a, b=b):
                actual = tuple(door_is_open(d, State("x", (a, b))) for d in doors)
                self.assertEqual(actual, (True, a, a and b, a and not b))

    def test_budget_exhaustion_is_failure_and_cannot_resume(self):
        game = MazeGame(example_task(), max_actions=8)
        for action in REFERENCE[:8]:
            result = game.step(action)
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.status, "failure")
        self.assertEqual(result.termination_reason, "action_limit")
        with self.assertRaises(RuntimeError):
            game.step(REFERENCE[-1])

    def test_unavailable_targets_count_but_malformed_actions_do_not_enter_engine(self):
        game = MazeGame(example_task(), max_actions=2)
        with self.assertRaises(TypeError):
            game.step(None)
        self.assertEqual(game.remaining_actions, 2)
        first = game.step(Action("move", "door_99"))
        second = game.step(Action("toggle", "control_1"))
        self.assertEqual((first.outcome, second.outcome), ("invalid_target", "invalid_target"))
        self.assertEqual(game.audit_state, example_task().initial)
        self.assertEqual(second.status, "failure")

    def test_episodes_are_independent(self):
        task = example_task()
        first, second = MazeGame(task), MazeGame(task)
        first.step(REFERENCE[0])
        first.step(REFERENCE[1])
        self.assertEqual(second.audit_state, task.initial)
        self.assertEqual(second.remaining_actions, 16)

    def test_public_feedback_has_no_hidden_state(self):
        result = asdict(MazeGame(example_task()).step(REFERENCE[0]))
        self.assertEqual(set(result), {"action", "outcome", "remaining_actions", "status", "reward", "termination_reason"})

    def test_invalid_budget_and_action_kind(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                MazeGame(example_task(), limit)
        with self.assertRaises(ValueError):
            Action("press", "door_1")


class SolverTests(unittest.TestCase):
    def test_shortest_solution_matches_independent_fixture(self):
        self.assertEqual(solve(example_task()), REFERENCE)
        report = analyze_task(example_task())
        self.assertEqual(report.solution_length, 9)
        self.assertEqual(report.minimum_toggles, 2)
        self.assertTrue(report.requires_revisit)
        self.assertFalse(report.dead_end_states)
        self.assertTrue(report.within_action_budget)
        self.assertLessEqual(report.reachable_state_count, 24)
        self.assertFalse(analyze_task(example_task(), max_actions=8).within_action_budget)

    def test_initial_switch_values_change_needed_actions(self):
        expected = {(False, False): (9, 2), (False, True): (6, 1),
                    (True, False): (6, 1), (True, True): (3, 0)}
        for bits, (length, toggles) in expected.items():
            with self.subTest(bits=bits):
                task = replace(example_task(), initial=State("Crescent", bits))
                report = analyze_task(task)
                self.assertEqual((report.solution_length, report.minimum_toggles), (length, toggles))
                self.assertEqual(report.requires_revisit, toggles > 0)

    def test_toggle_in_current_room_is_not_a_revisit(self):
        task = replace(example_task(), initial=State("Workshop", (False, True)))
        report = analyze_task(task)
        self.assertEqual(report.solution_length, 5)
        self.assertEqual(report.minimum_toggles, 1)
        self.assertFalse(report.requires_revisit)

    def test_unreachable_goal_is_detected(self):
        task = example_task()
        control = task.room("Tower").controls
        # Put B inside the Vault whose entrance already requires B on.
        rooms = tuple(replace(r, controls=()) if r.id == "Tower" else
                      replace(r, controls=control) if r.id == "Vault" else r for r in task.rooms)
        impossible = replace(task, rooms=rooms)
        self.assertIsNone(solve(impossible))
        report = analyze_task(impossible)
        self.assertIsNone(report.solution_length)
        self.assertIsNone(report.minimum_toggles)
        self.assertIsNone(report.requires_revisit)
        self.assertFalse(report.within_action_budget)
        self.assertEqual(len(report.dead_end_states), report.reachable_state_count)


if __name__ == "__main__":
    unittest.main()
