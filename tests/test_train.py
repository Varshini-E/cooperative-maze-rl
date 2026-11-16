"""Training-update preparation on synthetic trajectories (no model calls)."""
import unittest

try:
    from tinker_cookbook.rl.rollouts import do_single_rollout
    from tinker_cookbook.rl.types import TrajectoryGroup
    from panel_guide import train
    from panel_guide.scoring import prompt_action_scores, ratio_summary
    HAS_COOKBOOK = True
except ImportError:
    HAS_COOKBOOK = False

from tests.test_env import ACTION, ASK, MOVE, REPLY, Agent, Policy, make_env


def token_mass(data):
    values = [a for d in data for a in d.loss_fn_inputs['advantages'].to_torch().tolist()]
    return sum(a for a in values if a > 0), -sum(a for a in values if a < 0)


@unittest.skipUnless(HAS_COOKBOOK, "Install optional models dependencies to test training helpers.")
class PrepareUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def group(self):
        # One short success and one long failure, like most real groups.
        success = await do_single_rollout(Policy(MOVE), make_env(Agent(ACTION), exchanges=1))
        failure = await do_single_rollout(Policy(ASK, ASK, ASK), make_env(Agent(REPLY, REPLY, REPLY), exchanges=3))
        return TrajectoryGroup([success, failure], [0., 0.], [{}, {}])

    async def test_per_episode_balances_token_mass_and_token_sum_does_not(self):
        group = await self.group()
        data, masks, totals = train.prepare_update([group], 'token_sum')
        positive, negative = token_mass(data)
        self.assertGreater(negative / positive, 2)
        data, masks, totals = train.prepare_update([group], 'per_episode')
        positive, negative = token_mass(data)
        self.assertAlmostEqual(positive, negative, places=4)
        self.assertEqual((totals['mixed_groups'], totals['normalization']), (1, 'per_episode'))
        self.assertTrue(all('mask' not in d.loss_fn_inputs for d in data))
        for datum, mask in zip(data, masks):
            advantages = datum.loss_fn_inputs['advantages'].to_torch().tolist()
            self.assertTrue(all(a == 0 for a, m in zip(advantages, mask) if not m))

    async def test_uniform_groups_are_skipped_and_unknown_normalization_rejected(self):
        group = await self.group()
        uniform = TrajectoryGroup([group.trajectories_G[1]] * 2, [0., 0.], [{}, {}])
        data, masks, totals = train.prepare_update([uniform], 'per_episode')
        self.assertEqual((data, masks, totals['mixed_groups'], totals['episodes']), ([], [], 0, 2))
        with self.assertRaises(ValueError):
            train.prepare_update([group], 'mean_token')

    async def test_zero_progress_weight_matches_the_cookbook_episode_update(self):
        from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
        group = await self.group()
        data, _, _ = train.prepare_update([group], 'per_episode')
        scales = train.episode_scales(group, 'per_episode')
        expected, _ = assemble_training_data([group], [[float(a) * s for a, s in zip(compute_advantages([group])[0], scales)]])
        self.assertEqual([d.loss_fn_inputs['advantages'].to_torch().tolist() for d in data],
                         [d.loss_fn_inputs['advantages'].to_torch().tolist() for d in expected])

    async def test_progress_credit_trains_uniform_groups_on_moves_only(self):
        toward = await do_single_rollout(Policy(MOVE), make_env(Agent(ACTION), exchanges=1))
        away = await do_single_rollout(Policy(ASK, MOVE), make_env(Agent(REPLY, '{"type":"move","target":"door_1"}'), exchanges=2))
        uniform = TrajectoryGroup([toward, toward], [0., 0.], [{}, {}])
        self.assertEqual(train.prepare_update([uniform], 'per_episode')[0], [])
        data, masks, totals = train.prepare_update([uniform], 'per_episode', progress_weight=.5)
        self.assertEqual(len(data), 2)
        for datum, mask in zip(data, masks):
            self.assertEqual(datum.loss_fn_inputs['advantages'].to_torch().tolist(), [.5 * m for m in mask])
        # Mixed group: the success's move gets A + w; the failure's question gets only its share of A
        # and its move into the wrong room gets A - w, after per-episode scaling.
        group = TrajectoryGroup([toward, away], [0., 0.], [{}, {}])
        data, masks, totals = train.prepare_update([group], 'per_episode', progress_weight=.5)
        s = train.episode_scales(group, 'per_episode')
        firsts = [next(a for a, m in zip(d.loss_fn_inputs['advantages'].to_torch().tolist(), mask) if m)
                  for d, mask in zip(data, masks)]
        for value, expected in zip(firsts, (s[0] * 1., s[1] * -.5, s[1] * -1.), strict=True):
            self.assertAlmostEqual(value, expected, places=6)
        self.assertEqual(totals['progress_turns'], {'closer': 1, 'farther': 1, 'neutral': 1})
        with self.assertRaises(ValueError):
            train.prepare_update([group], 'per_episode', progress_weight=-1)

    async def test_alignment_check_counts_expert_tokens_only(self):
        group = await self.group()
        checked = train.verify_training_alignment(group.trajectories_G[0])
        self.assertEqual(checked['datums'], 1)
        self.assertEqual(checked['expert_target_tokens'], len(MOVE) + 1)


@unittest.skipUnless(HAS_COOKBOOK, "Install optional models dependencies to test scoring helpers.")
class ScoringTests(unittest.TestCase):
    def test_identical_scores_give_unit_ratios(self):
        summary = ratio_summary([-.5, -1., -2.], [-.5, -1., -2.])
        self.assertEqual((summary['ratio_min'], summary['ratio_max'], summary['token_ess_fraction']), (1., 1., 1.))

    def test_ratios_are_tokenwise_and_misalignment_is_rejected(self):
        summary = ratio_summary([0., -.1], [-.1, 0.])
        self.assertAlmostEqual(summary['mean_log_ratio'], 0.)
        self.assertAlmostEqual(summary['max_abs_log_ratio'], .1)
        for training, sampling in (([0.], [0., 0.]), ([], []), ([float('nan')], [0.])):
            with self.assertRaises(ValueError):
                ratio_summary(training, sampling)
        self.assertEqual(prompt_action_scores([None, -1., -2., -3.], [0, 1, 1]), [-2., -3.])


if __name__ == '__main__':
    unittest.main()
