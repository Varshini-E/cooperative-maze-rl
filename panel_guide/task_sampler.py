"""Pick training tasks that are likely to produce mixed-reward groups.

A group of G episodes trains only if its rewards disagree. With success rate p that happens
with probability 1 - p^G - (1-p)^G: ~0.99 at p=0.5, ~0.57 at p=0.9, 0 at p=0 or 1. Instead of
DAPO-style oversampling (pay for groups, then discard uniform ones), keep a per-task belief
about the *current* policy's p and spend rollouts where that probability is high.

Belief: Beta(a0 + s, b0 + f). The prior (a0, b0) comes from the route-length band; s and f are
successes/failures from this run's own training groups, multiplied by `decay` after every
optimizer update so evidence about older policies fades. No extra model calls are needed.
"""

import math
import random
from collections import defaultdict

# Untrained thinking policy on development tasks (reasoning-comparison-01 + token-budget-01):
# routes 1-2: 11/12, 3-4: 9/12, 5-6: 5/12. Routes of 7+ actions were never measured; 0.25 is a guess.
DEFAULT_PRIOR = ((2, .9), (4, .75), (6, .42), (10, .25))


def prior_mean(solution_length, prior=DEFAULT_PRIOR):
    for upper, mean in prior:
        if solution_length <= upper:
            return mean
    raise ValueError('Solution length exceeds the prior table.')


def beta_moment(a, b, g):
    """E[p^g] for p ~ Beta(a, b)."""
    return math.prod((a + i) / (a + b + i) for i in range(g))


def mixed_probability(a, b, g):
    """Posterior-averaged chance that g independent episodes do not all agree."""
    return max(0., 1. - beta_moment(a, b, g) - beta_moment(b, a, g))


class TaskSampler:
    def __init__(self, rows, group_size=8, prior=DEFAULT_PRIOR, prior_strength=2., decay=.8,
                 floor=.02, seed=0):
        if not rows or group_size < 2 or prior_strength <= 0 or not 0 < decay <= 1 or floor < 0:
            raise ValueError('Need tasks, group_size >= 2, positive prior strength, decay in (0, 1].')
        self.rows = {r['task_id']: r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError('Duplicate task IDs.')
        self.group_size, self.decay, self.floor = group_size, decay, floor
        self.prior = {i: (prior_strength * prior_mean(r['solution_length'], prior),
                          prior_strength * (1 - prior_mean(r['solution_length'], prior)))
                      for i, r in self.rows.items()}
        self.evidence = defaultdict(lambda: [0., 0.])
        self.uses = defaultdict(int)
        self.rng = random.Random(seed)

    def posterior(self, task_id):
        (a, b), (s, f) = self.prior[task_id], self.evidence[task_id]
        return a + s, b + f

    def success_estimate(self, task_id):
        a, b = self.posterior(task_id)
        return a / (a + b)

    def p_mixed(self, task_id):
        return mixed_probability(*self.posterior(task_id), self.group_size)

    def sample(self, n):
        """Weighted by p_mixed without replacement, at most one task per maze per batch.

        `floor` keeps every task reachable so a stale 'always solved' belief can be revisited.
        """
        candidates = list(self.rows)
        if len({self.rows[i]['maze_id'] for i in candidates}) < n:
            raise ValueError('Fewer distinct mazes than requested tasks.')
        chosen, mazes = [], set()
        while len(chosen) < n:
            pool = [i for i in candidates if self.rows[i]['maze_id'] not in mazes]
            weights = [self.p_mixed(i) + self.floor for i in pool]
            pick = self.rng.choices(pool, weights)[0]
            chosen.append(pick)
            mazes.add(self.rows[pick]['maze_id'])
            candidates.remove(pick)
        for i in chosen:
            self.uses[i] += 1
        return [self.rows[i] for i in chosen]

    def observe(self, task_id, rewards):
        if task_id not in self.rows or not rewards or any(r not in (0, 1) for r in rewards):
            raise ValueError('Observe binary rewards for a known task.')
        self.evidence[task_id][0] += sum(rewards)
        self.evidence[task_id][1] += len(rewards) - sum(rewards)

    def advance(self):
        """Call after each optimizer update: older evidence describes an older policy."""
        for counts in self.evidence.values():
            counts[0] *= self.decay
            counts[1] *= self.decay

    def state(self):
        return {'evidence': {k: list(v) for k, v in self.evidence.items() if any(v)},
                'uses': dict(self.uses), 'rng': self.rng.getstate()}

    def load(self, state):
        for k, v in state['evidence'].items():
            self.evidence[k] = list(v)
        self.uses.update(state['uses'])
        version, internal, gauss = state['rng']
        self.rng.setstate((version, tuple(internal), gauss))
