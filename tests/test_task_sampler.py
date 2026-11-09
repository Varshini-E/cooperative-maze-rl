import json
import random
import unittest
from collections import Counter

from panel_guide.task_sampler import TaskSampler, beta_moment, mixed_probability


class MathTests(unittest.TestCase):
    def test_beta_moment_matches_monte_carlo(self):
        rng = random.Random(0)
        draws = [rng.betavariate(2., 3.) for _ in range(200000)]
        self.assertAlmostEqual(beta_moment(2., 3., 8), sum(p ** 8 for p in draws) / len(draws), places=3)

    def test_mixed_probability_extremes(self):
        self.assertGreater(mixed_probability(50., 50., 8), .98)
        self.assertLess(mixed_probability(200., .01, 8), .01)
        # A vague prior is less sure of a mixed group than a confident p = 0.5.
        self.assertLess(mixed_probability(1., 1., 8), mixed_probability(50., 50., 8))


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [dict(task_id=f'm{maze}-{length}', maze_id=f'm{maze}', solution_length=length)
                    for maze in range(12) for length in range(1, 11)]

    def test_one_task_per_maze_and_deterministic(self):
        a, b = TaskSampler(self.rows, seed=4), TaskSampler(self.rows, seed=4)
        batch = a.sample(8)
        self.assertEqual([r['task_id'] for r in batch], [r['task_id'] for r in b.sample(8)])
        self.assertEqual(len({r['maze_id'] for r in batch}), 8)

    def test_prior_prefers_longer_routes_over_near_certain_ones(self):
        sampler = TaskSampler(self.rows, seed=1)
        short = next(r for r in self.rows if r['solution_length'] == 1)['task_id']
        mid = next(r for r in self.rows if r['solution_length'] == 5)['task_id']
        self.assertGreater(sampler.p_mixed(mid), sampler.p_mixed(short))

    def test_saturated_tasks_fade_and_decay_restores_uncertainty(self):
        sampler = TaskSampler(self.rows, seed=2)
        task = next(r for r in self.rows if r['solution_length'] == 5)['task_id']
        before = sampler.p_mixed(task)
        sampler.observe(task, [1] * 8)
        sampler.observe(task, [1] * 8)
        solved = sampler.p_mixed(task)
        self.assertLess(solved, .5 * before)
        for _ in range(10):
            sampler.advance()
        self.assertGreater(sampler.p_mixed(task), solved)

    def test_observed_uniform_tasks_are_drawn_less_often(self):
        solved = {r['task_id'] for r in self.rows if r['solution_length'] >= 5}

        def share(observe):
            sampler = TaskSampler(self.rows, seed=3)
            if observe:
                for task in solved:
                    sampler.observe(task, [1] * 16)
            drawn = Counter(r['task_id'] in solved for _ in range(40) for r in sampler.sample(8))
            return drawn[True] / sum(drawn.values())
        self.assertLess(share(True), .85 * share(False))

    def test_state_round_trip(self):
        sampler = TaskSampler(self.rows, seed=5)
        sampler.observe(self.rows[0]['task_id'], [0, 1, 1, 0])
        sampler.sample(4)
        clone = TaskSampler(self.rows, seed=99)
        clone.load(json.loads(json.dumps(sampler.state())))
        self.assertEqual([r['task_id'] for r in clone.sample(6)], [r['task_id'] for r in sampler.sample(6)])

    def test_rejects_non_binary_rewards(self):
        with self.assertRaises(ValueError):
            TaskSampler(self.rows).observe(self.rows[0]['task_id'], [0.5])


if __name__ == '__main__':
    unittest.main()
