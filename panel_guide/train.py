"""Expert RL update helpers: group-centred advantages (plus optional per-turn progress credit).

Only Expert action tokens carry advantages; the manual, Operator replies and feedback are context.
One importance-sampling step per batch by default; `optimize` runs several clipped `ppo` steps. No reference KL.
"""

import json
import math

import tinker
from tinker.lib.retry_handler import RetryConfig
from tinker_cookbook.rl.data_processing import compute_advantages, trajectory_to_data


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_training_alignment(trajectory) -> dict:
    # Nonzero test advantage makes masking errors visible. This is not a training group.
    data = trajectory_to_data(trajectory, traj_advantage=1.0)
    _require(len(data) == len(trajectory.transitions), "Expected one datum per rebuilt Expert prompt.")
    sampled_tokens, context_tokens = 0, 0
    for transition, datum in zip(trajectory.transitions, data):
        observation, action = transition.ob.to_ints(), transition.ac.tokens
        full = observation + action
        fields = {name: tensor.to_torch().tolist() for name, tensor in datum.loss_fn_inputs.items()}
        _require(datum.model_input.to_ints() == full[:-1], "Model input is not correctly shifted.")
        _require(fields["target_tokens"] == full[1:], "Next-token targets are misaligned.")
        expected_mask = [0.] * (len(observation) - 1) + [1.] * len(action)
        _require(fields["mask"] == expected_mask, "Non-Expert tokens receive loss or Expert tokens are lost.")
        _require(fields["advantages"] == expected_mask, "Advantages include context tokens.")
        expected_logprobs = [0.] * (len(observation) - 1) + transition.ac.logprobs
        _require(len(fields["logprobs"]) == len(expected_logprobs) and all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(fields["logprobs"], expected_logprobs)),
            "Sampling logprobs were altered or shifted incorrectly.")
        sampled_tokens += len(action)
        context_tokens += len(observation) - 1
    return {"datums": len(data), "expert_target_tokens": sampled_tokens, "masked_context_positions": context_tokens}


NORMALIZATIONS = ('token_sum', 'per_episode')


def episode_scales(group, normalization):
    """Per-trajectory advantage multipliers.

    token_sum:   1 for every episode; each sampled token carries the full advantage,
                 so long (usually failed) episodes dominate the summed loss.
    per_episode: mean_T / T_i over the group's Expert action tokens, so every
                 episode carries equal total weight and, because group-centred
                 advantages sum to zero, positive and negative token mass balance.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f'normalization must be one of {NORMALIZATIONS}.')
    lengths = [sum(len(t.ac.tokens) for t in trajectory.transitions) for trajectory in group.trajectories_G]
    if any(n == 0 for n in lengths):
        raise ValueError('Every trajectory needs at least one Expert action token.')
    mean = sum(lengths) / len(lengths)
    return [mean / n if normalization == 'per_episode' else 1. for n in lengths]


def turn_advantages(groups, normalization, progress_weight):
    """Per-group, per-episode, per-Expert-turn advantages.

    a[i][t] = scale_i * (A_i + progress_weight * progress_it), where A_i is the group-centred goal
    reward, scale_i the episode normalization and progress_it the Env's solver-verified progress
    metric for that turn (+1 closer, -1 farther, 0 otherwise). The progress term needs no baseline:
    its sign is correct in every state, so groups with one shared outcome still carry signal.
    """
    result = []
    for group, group_advantages in zip(groups, compute_advantages(groups)):
        scales = episode_scales(group, normalization)
        result.append([[scale * (float(a) + progress_weight * t.metrics.get('progress', 0))
                        for t in trajectory.transitions]
                       for trajectory, a, scale in zip(group.trajectories_G, group_advantages, scales)])
    return result


def assemble_turn_data(groups, advantages):
    """One datum per rebuilt Expert prompt with its own advantage; all-zero datums are dropped
    (they add no gradient, only forward-backward cost)."""
    data, provenance = [], []
    for g, (group, group_advantages) in enumerate(zip(groups, advantages)):
        for i, (trajectory, turns) in enumerate(zip(group.trajectories_G, group_advantages)):
            datums = trajectory_to_data(trajectory, 1.0)  # advantages == mask; checked by verify_training_alignment
            for t, (datum, value) in enumerate(zip(datums, turns, strict=True)):
                if value == 0:
                    continue
                mask = datum.loss_fn_inputs['mask'].to_torch()
                datum.loss_fn_inputs['advantages'] = tinker.TensorData.from_torch(mask * value)
                data.append(datum)
                provenance.append({'group_idx': g, 'traj_idx': i, 'turn_idx': t})
    return data, provenance


def prepare_update(groups, normalization, progress_weight=0.0):
    """Keep all outcomes on disk; omit only zero-advantage groups (and turns) from compute.

    progress_weight = 0 is the sparse-reward update used by every run so far: only groups with
    mixed outcomes train. A positive weight adds per-turn solver progress credit (turn_advantages).
    """
    if not math.isfinite(progress_weight) or progress_weight < 0:
        raise ValueError('progress_weight must be finite and nonnegative.')
    totals = {'episodes': 0, 'datums': 0, 'expert_target_tokens': 0, 'masked_context_positions': 0}
    for group in groups:
        for trajectory in group.trajectories_G:
            checked = verify_training_alignment(trajectory)
            totals['episodes'] += 1
            for key, value in checked.items():
                totals[key] += value
    informative = [g for g in groups if len(set(g.get_total_rewards())) > 1]
    totals['mixed_groups'] = len(informative)
    if progress_weight:
        progress = [t.metrics.get('progress', 0) for g in groups for tr in g.trajectories_G for t in tr.transitions]
        totals.update(progress_weight=progress_weight, progress_turns={
            'closer': progress.count(1), 'farther': progress.count(-1), 'neutral': progress.count(0)})
        trained = groups
    else:
        trained = informative
    if not trained:
        return [], [], totals
    totals['normalization'] = normalization
    data, provenance = assemble_turn_data(trained, turn_advantages(trained, normalization, progress_weight))
    if not data:
        return [], [], totals
    masks = [d.loss_fn_inputs['mask'].to_torch().tolist() for d in data]
    for datum, mask in zip(data, masks):
        advantages = datum.loss_fn_inputs['advantages'].to_torch().tolist()
        if any(a != 0 for a, m in zip(advantages, mask) if m == 0):
            raise ValueError('Context received a training advantage.')
    totals.update(training_datums=len(data), training_input_tokens=sum(d.model_input.length for d in data),
                  training_target_tokens=sum(int(sum(mask)) for mask in masks), provenance=provenance)
    # Tinker's RL loss uses zero advantages to mask context. `mask` is Cookbook audit metadata.
    payload = [tinker.Datum(model_input=d.model_input,
               loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != 'mask'}) for d in data]
    return payload, masks, totals


def action_logprobs(result, index, mask):
    values = result.loss_fn_outputs[index]['logprobs'].to_torch().tolist()
    if len(values) != len(mask):
        raise ValueError('Training log probabilities are misaligned.')
    selected = [v for v, enabled in zip(values, mask) if enabled]
    if not selected or not all(math.isfinite(v) for v in selected):
        raise ValueError('Nonfinite or missing training log probabilities.')
    return selected


def log_ratios(data, masks, result):
    """Current (training) minus sampling log-probabilities on Expert action tokens."""
    values = []
    for index, (datum, mask) in enumerate(zip(data, masks, strict=True)):
        sampled = [v for v, m in zip(datum.loss_fn_inputs['logprobs'].to_torch().tolist(), mask) if m]
        values += [c - s for c, s in zip(action_logprobs(result, index, mask), sampled, strict=True)]
    return values


def ratio_stats(values, clip=None):
    stats = {'mean_abs_log_ratio': sum(map(abs, values)) / len(values),
             'mean_log_ratio': sum(values) / len(values)}
    if clip:
        low, high = clip
        stats['clip_fraction'] = sum(not low <= math.exp(v) <= high for v in values) / len(values)
    return stats


LOSSES = ('importance_sampling', 'ppo')


async def optimize(client, data, masks, settings):
    """`steps_per_batch` forward-backward + Adam steps on one sampled batch.

    Every step compares against the *sampling* log-probabilities, so after the first step the batch
    is off-policy; more than one step therefore requires the clipped `ppo` loss. Each step's stats
    describe the policy before that step (step 1: sampler/trainer mismatch only).
    """
    steps, loss_fn = settings.get('steps_per_batch', 1), settings['loss_fn']
    if loss_fn not in LOSSES or type(steps) is not int or steps < 1:
        raise ValueError(f'loss_fn must be one of {LOSSES} and steps_per_batch a positive integer.')
    if steps > 1 and loss_fn != 'ppo':
        raise ValueError("Several steps per batch reuse off-policy samples; use loss_fn = 'ppo'.")
    clip = tuple(settings['ppo_clip']) if loss_fn == 'ppo' else None
    config = {'clip_low_threshold': clip[0], 'clip_high_threshold': clip[1]} if clip else None
    adam = tinker.AdamParams(learning_rate=settings['learning_rate'], beta1=.9, beta2=.95, eps=1e-8,
                             grad_clip_norm=settings['grad_clip_norm'])
    history = []
    for step in range(1, steps + 1):
        backward = await (await client.forward_backward_async(data, loss_fn=loss_fn, loss_fn_config=config)).result_async()
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in backward.metrics.values()):
            raise ValueError('Nonfinite backward metric; no optimizer step.')
        optimizer = await (await client.optim_step_async(adam)).result_async()
        history.append({'step': step, **backward.metrics, **optimizer.metrics,
                        **ratio_stats(log_ratios(data, masks, backward), clip)})
    return history


async def save_checkpoint(client, name, ttl):
    state = await (await client.save_state_async(name, ttl_seconds=ttl)).result_async()
    sample = await (await client.save_weights_for_sampler_async(name, ttl_seconds=ttl)).result_async()
    return {'state_path': state.path, 'sampler_path': sample.path, 'ttl_seconds': ttl}


async def attach_sampler(service, expert, checkpoint):
    sampler = await service.create_sampling_client_async(model_path=checkpoint['sampler_path'],
                      retry_config=RetryConfig(enable_retry_logic=False))
    expert.sampler = sampler
    expert.model_path = checkpoint['sampler_path']
    expert.policy_id = checkpoint['sampler_path']
    return sampler
