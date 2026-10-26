import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from panel_guide.demo import run_demo
from panel_guide.engine import MazeGame, RoomView
from panel_guide.render import render_room
from panel_guide.tasks import SYMBOLS, State, example_task


class DemoTests(unittest.TestCase):
    def test_demo_outputs_success_recovery_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            audit = run_demo(output)
            saved = json.loads((output / "audit.json").read_text())
            self.assertEqual(saved["kind"], "privileged_maze_demo_trace")
            self.assertEqual(len(saved["solution"]), 9)
            outcomes = [(e["status"], e["action_attempts"], e["total_reward"]) for e in audit["episodes"]]
            self.assertEqual(outcomes, [("success", 9, 1), ("success", 12, 1), ("failure", 16, 0)])
            recovery = saved["episodes"][1]["events"]
            self.assertEqual(recovery[1]["feedback"]["outcome"], "blocked")
            self.assertEqual(recovery[1]["before"], recovery[1]["after"])
            self.assertTrue((output / "manual.txt").is_file())
            for episode in saved["episodes"]:
                observations = json.loads((output / episode["name"] / "observations.json").read_text())
                self.assertEqual(len(observations), episode["action_attempts"] + 1)
                for observation in observations:
                    self.assertEqual(set(observation["view"]), {"emblem", "doors", "controls"})
                    with Image.open(output / observation["image"]) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertEqual(image.size, (1480, 852))
                        self.assertFalse(image.info)
                    with Image.open(output / observation["detail_image"]) as detail:
                        self.assertEqual(detail.format, "PNG")
                        self.assertFalse(detail.info)
            # A rerun with a smaller limit must not leave old generated frames.
            run_demo(output, max_actions=2)
            self.assertEqual(len(list((output / "reference").glob("frame-*.png"))), 3)
            self.assertEqual(len(list((output / "reference").glob("detail-*.png"))), 3)

    def test_hidden_state_does_not_change_identical_room_images(self):
        task = example_task()
        other = replace(task, initial=State("Crescent", (True, True)), goal="Tower")
        left_view, right_view = MazeGame(task).view, MazeGame(other).view
        self.assertEqual(left_view, right_view)
        with render_room(left_view) as left, render_room(right_view) as right:
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_all_symbols_have_distinct_renderings(self):
        pixels = set()
        for symbol in SYMBOLS:
            with render_room(RoomView(symbol, (), ())) as image:
                pixels.add(image.tobytes())
        self.assertEqual(len(pixels), len(SYMBOLS))


if __name__ == "__main__":
    unittest.main()
