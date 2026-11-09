"""Expert RL update helpers: group-centred advantages, one importance-sampling step per batch.

Only Expert action tokens carry advantages; the manual, Operator replies and feedback are context.
There is no PPO clipping or reference KL.
"""

import json
import math

import tinker
from tinker.lib.retry_handler import RetryConfig
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages, trajectory_to_data


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


def prepare_update(groups, normalization):
    """Keep all outcomes on disk; omit only zero-advantage groups from compute."""
    totals = {'episodes': 0, 'datums': 0, 'expert_target_tokens': 0, 'masked_context_positions': 0}
    for group in groups:
        for trajectory in group.trajectories_G:
            checked = verify_training_alignment(trajectory)
            totals['episodes'] += 1
            for key, value in checked.items():
                totals[key] += value
    informative = [g for g in groups if len(set(g.get_total_rewards())) > 1]
    totals['mixed_groups'] = len(informative)
    if not informative:
        return [], [], totals
    advantages = [[float(a) * scale for a, scale in zip(group_advantages, episode_scales(group, normalization))]
                  for group, group_advantages in zip(informative, compute_advantages(informative))]
    totals['normalization'] = normalization
    data, provenance = assemble_training_data(informative, advantages)
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
