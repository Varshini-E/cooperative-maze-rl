import io
import json
import unittest
from dataclasses import asdict, replace
from itertools import product

from PIL import Image

from panel_guide.conversation import Conversation
from panel_guide.curriculum import development_task
from panel_guide.engine import Action, MazeGame, observe, observe_maze
from panel_guide.render import render_maze, render_room
from panel_guide.tasks import State, example_task
from panel_guide.tinker_backend import model_messages


class MazeViewTests(unittest.TestCase):
    def test_geometry_matches_every_passage_and_only_current_state_is_exposed(self):
        task = example_task()
        for room in task.rooms:
            for bits in product((False, True), repeat=2):
                state = State(room.id, bits)
                view = observe_maze(task, state)
                self.assertEqual(len(view.rooms), 6)
                self.assertEqual(sum(r.current for r in view.rooms), 1)
                for source, visible in zip(task.rooms, view.rooms, strict=True):
                    self.assertEqual(visible.position, source.position)
                    self.assertEqual(visible.emblem, source.emblem)
                    self.assertEqual([d.neighbor for d in visible.doors],
                                     [task.room(d.destination).position for d in source.doors])
                    if visible.current:
                        local = observe(task, state)
                        self.assertEqual([(d.id, d.symbol, d.is_open) for d in visible.doors],
                                         [(d.id, d.symbol, d.is_open) for d in local.doors])
                        self.assertEqual([(c.id, c.symbol, c.is_on) for c in visible.controls],
                                         [(c.id, c.symbol, c.is_on) for c in local.controls])
                    else:
                        self.assertTrue(all(d.id is None and d.is_open is None for d in visible.doors))
                        self.assertTrue(all(c.id is None and c.is_on is None for c in visible.controls))
                serialized = json.dumps(asdict(view))
                for private in ('"goal"', '"requirements"', '"switch"', '"solution"', '"Vault"'):
                    self.assertNotIn(private, serialized)

    def test_hidden_remote_values_and_private_goal_do_not_change_pixels(self):
        task = example_task()
        other = replace(task, goal="Tower")
        left = observe_maze(task, State("Crescent", (False, False)))
        right = observe_maze(other, State("Crescent", (True, True)))
        self.assertEqual(left, right)
        with render_maze(left) as a, render_maze(right) as b:
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_local_gate_changes_and_visited_remote_states_are_hidden_again(self):
        task = example_task()
        closed = observe_maze(task, State("Garden", (False, False)))
        opened = observe_maze(task, State("Garden", (True, False)))
        with render_maze(closed) as a, render_maze(opened) as b:
            self.assertNotEqual(a.tobytes(), b.tobytes())
        game = MazeGame(task)
        game.step(Action("move", "door_2"))
        game.step(Action("toggle", "control_1"))
        workshop = next(r for r in game.maze_view.rooms if r.current)
        self.assertTrue(workshop.controls[0].is_on)
        game.step(Action("move", "door_1"))
        workshop = next(r for r in game.maze_view.rooms if r.emblem == "anvil")
        self.assertIsNone(workshop.controls[0].is_on)
        self.assertIsNone(workshop.controls[0].id)

    def test_visible_remote_controls_cannot_be_used_from_another_room(self):
        game = MazeGame(example_task())
        self.assertEqual(sum(len(r.controls) for r in game.maze_view.rooms), 2)
        before = game.audit_state
        result = game.step(Action("toggle", "control_1"))
        self.assertEqual(result.outcome, "invalid_target")
        self.assertEqual(game.audit_state, before)

    def test_slot_variants_keep_the_same_floor_plan(self):
        plans = []
        target_orders = set()
        for layout in range(6):
            view = observe_maze(development_task("one-door", layout), State("Gallery", (True, True)))
            plans.append({r.emblem: (r.position, frozenset(d.neighbor for d in r.doors)) for r in view.rooms})
            current = next(r for r in view.rooms if r.current)
            target_orders.add(tuple((d.symbol, d.id) for d in current.doors))
        self.assertTrue(all(p == plans[0] for p in plans))
        self.assertEqual(len(target_orders), 3)

    def test_invalid_floor_plans_are_rejected(self):
        task = example_task()
        for position in (task.rooms[0].position, (5, 5)):
            rooms = tuple(replace(r, position=position) if r.id == "Workshop" else r for r in task.rooms)
            with self.assertRaises(ValueError):
                replace(task, rooms=rooms)
        for position in ((-1, 0), (0, True), (0,)):
            with self.assertRaises(ValueError):
                replace(task.rooms[0], position=position)

    def test_operator_gets_one_full_maze_image_and_expert_remains_text_only(self):
        conversation = Conversation(example_task(), prompt_version="v3")
        conversation.history = [{"role": "expert", "type": "ask", "text": "Describe your current room."}]
        request = conversation._request("operator")
        with Image.open(io.BytesIO(request.image_png)) as actual, render_maze(conversation.game.maze_view) as expected:
            self.assertEqual(actual.size, (1480, 852))
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertFalse(actual.info)
        messages = model_messages(request)
        parts = messages[-1]["content"]
        self.assertEqual([p["type"] for p in parts], ["image", "text"])
        self.assertNotIn("PRIVATE MANUAL", str(messages))
        self.assertNotIn("maze_view", str(messages))
        self.assertIn("gray ? means UNKNOWN", request.system)
        expert = conversation._request("expert")
        self.assertIsNone(expert.image_png)
        self.assertIsNone(expert.scripted_view)
        self.assertIn("PRIVATE MANUAL", expert.system)

    def test_legacy_room_mode_is_explicit_and_still_available(self):
        conversation = Conversation(example_task(), observation_mode="room")
        request = conversation._request("operator")
        with Image.open(io.BytesIO(request.image_png)) as actual, render_room(conversation.game.view) as expected:
            self.assertEqual(actual.size, (960, 720))
            self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertNotIn("ONE complete maze", request.system)
        with self.assertRaises(ValueError):
            Conversation(example_task(), observation_mode="invalid")


if __name__ == "__main__":
    unittest.main()
