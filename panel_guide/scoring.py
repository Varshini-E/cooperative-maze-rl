"""Compare the exact token probabilities used by the RL objective."""

import math

import numpy as np


def ratio_summary(training, sampling):
    """Tokenwise exp(log p_train - log p_sample), never a sequence product."""
    p, q = np.asarray(training, dtype=float), np.asarray(sampling, dtype=float)
    if p.ndim != 1 or p.shape != q.shape or not p.size:
        raise ValueError('Scoring needs equal, nonempty token vectors.')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite token scores.')
    delta = p - q
    with np.errstate(over='ignore', under='ignore'):
        ratios = np.exp(delta)
    if not np.isfinite(ratios).all() or np.any(ratios == 0):
        raise ValueError('Nonfinite or underflowed importance ratios.')
    normalized = ratios / ratios.max()
    return {'tokens': int(p.size), 'mean_log_ratio': float(delta.mean()),
            'mean_abs_log_ratio': float(np.abs(delta).mean()),
            'max_abs_log_ratio': float(np.abs(delta).max()),
            'ratio_min': float(ratios.min()), 'ratio_max': float(ratios.max()),
            'ratio_mean': float(ratios.mean()),
            'ratio_p01': float(np.quantile(ratios, .01)),
            'ratio_p99': float(np.quantile(ratios, .99)),
            'fraction_outside_0_8_1_2': float(((ratios < .8) | (ratios > 1.2)).mean()),
            'fraction_outside_0_5_2': float(((ratios < .5) | (ratios > 2)).mean()),
            'token_ess_fraction': float(normalized.sum() ** 2 / (p.size * (normalized ** 2).sum()))}


def selected_scores(values, mask):
    if len(values) != len(mask):
        raise ValueError('Scoring mask is misaligned.')
    selected = [value for value, enabled in zip(values, mask) if enabled]
    if not selected or any(value is None or not math.isfinite(value) for value in selected):
        raise ValueError('Missing or nonfinite action scores.')
    return selected


def prompt_action_scores(prompt_scores, mask):
    """Training position i predicts prompt position i+1, even with context gaps."""
    if len(prompt_scores) != len(mask) + 1:
        raise ValueError('Prompt scoring must include the full shifted sequence.')
    return selected_scores(prompt_scores[1:], mask)


def batch_ratios(data, masks, outputs):
    if not len(data) == len(masks) == len(outputs):
        raise ValueError('Training output count differs from the batch.')
    training, sampling = [], []
    for datum, mask, output in zip(data, masks, outputs):
        training.extend(selected_scores(output['logprobs'].to_torch().tolist(), mask))
        sampling.extend(selected_scores(datum.loss_fn_inputs['logprobs'].to_torch().tolist(), mask))
    return ratio_summary(training, sampling)
