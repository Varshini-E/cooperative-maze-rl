# RL status — November 9, 2026

Expert-only training (Qwen3-8B LoRA, thinking enabled); the Operator is frozen. Each task gets 8 episodes; advantage = reward − group mean; one importance-sampling step per batch, without PPO clipping or KL.

| Experiment | Result |
| --- | --- |
| Original RL, token-summed loss, LR 4.7e-4 | Format collapse (~60% rejected messages) |
| LR 1e-6, two easy tasks | 12/16 → 8/16 |
| Offline audit | Failed episodes got 2.2–6.8× more loss weight than successes |
| Overfit test | Per-episode normalization at LR 1e-4 raises successes and lowers failures |
| Private Expert notes (A/B) | 19/24 vs 21/24, no detectable effect |
| 4-task pilot | 20/32 → 25/32; hardest task 2/8 → 7/8 |
| Held-out check | 37 vs 36 of 48 (pilot vs initial) |
| Scripted correct Expert + real Operator | 15/16 |
| Broader run, ~24 tasks, 4 updates | 81 vs 82 of 120 held-out; 95% CI −10.9 to +8.3 points |

**Conclusion:** RL fits the tasks it trains on but has not generalized. Each update moved token log-probabilities by ~0.016, about the sampler/trainer noise level, and four updates are too few. The Expert's planning, not the Operator, limits success.

**Next:** improve the Expert's state tracking (structured notes or room feedback), then try more optimization per batch (2–3 clipped steps or a higher LR).
