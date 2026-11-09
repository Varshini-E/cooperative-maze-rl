import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from panel_guide.demo import run_demo
from panel_guide.render import draw_symbol
from panel_guide.tasks import SYMBOLS


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

    def test_all_symbols_have_distinct_renderings(self):
        pixels = set()
        for symbol in SYMBOLS:
            image = Image.new("RGB", (80, 80), "white")
            draw_symbol(ImageDraw.Draw(image), symbol, (40, 40), 32)
            pixels.add(image.tobytes())
        self.assertEqual(len(pixels), len(SYMBOLS))

if __name__ == "__main__":
    unittest.main()
