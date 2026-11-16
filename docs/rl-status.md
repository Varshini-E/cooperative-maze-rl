# RL status — November 16, 2026

Only the Expert (Qwen3-8B LoRA, thinking enabled) is trained, and the frozen Operator (Qwen3.5-9B) plays in every training and evaluation rollout. Each task gets 8 episodes. The advantage is reward minus the group mean, normalized per episode. Each batch gets one importance-sampling step or three clipped PPO steps, with no reference KL.

## Results

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
| Sampler vs trainer log-probs | Mean ratio 1.000, effective sample size (ESS) 0.998: on-policy |
| 3 clipped PPO steps per batch | Success-token log-probs move ~9× more than with one step; ≤2% clipped |
| Manual hint at 36 wrong decisions | +4.9 points correct (CI −2.8 to +12.5); rejected |
| Restart-point RL (132 mid-episode starts, 4 updates) | 42 vs 41 of 85 held-out starts (initial vs trained) |
| Decision accuracy of that policy | Its own training starts: 35.2% → 35.8% |
| Overfit 4 decisions (10 updates, LR 3e-4, progress credit) | Correct first message 7.8% → 17.2% (target ≥50%) |
| Fine-tuning on 273 solver-checked Expert turns | Held-out first move closer to the goal: 36.8% → 38.2% (CI −4.4 to +7.8); full episodes 42 → 49 of 85 (p = 0.30) |

## What we learned

- **RL fits, but does not transfer.** Training tasks improve, while unseen tasks have not moved in any run.
- **The Expert limits success.** A scripted correct Expert with the real Operator solves 15/16. After a correct first move the success rate is 74%, against 40% after a wrong one.
- **Errors are systematic.** 95% of wrong choices repeat the same wrong door, and 16 of 36 decisions were never correct in 8 samples. More sampling rarely finds the fix.
- **Updates barely move the policy.** One update shifts token log-probs by ~0.016, about the sampler/trainer noise. Even 10 updates on 4 decisions barely moved them, and fine-tuning hardly changed its own training points (45.5% → 46.6%). So far the limit is optimization strength, not proven model capacity.
- **Episode success is a noisy metric.** About 40% of paired restarts disagree, so 85 pairs detect only effects of roughly 15 points or more. Accuracy of the first move, scored by the solver with the live Operator, is cheaper and sharper.
- **Check what a metric counts.** Scoring only the first message counted questions as wrong answers. Scoring the first physical move fixes this.
- **Budget actual tokens, not the worst case.** The spending guard reserved ~2× the real cost until reservations were settled after each call.

## Tools added

- Mid-episode restarts (`train_rl --decisions`)
- Solver progress credit (`progress_weight`)
- Early stop for provably lost episodes (`stop_when_lost`, which saves ~19% of tokens)
- Clipped PPO steps (`steps_per_batch`)
- Settled spending reservations

## Next

1. Fine-tune harder on the collected data (more epochs or a higher LR) until first-move accuracy on the training points clearly rises, then re-test on held-out points.
2. Run a larger Expert, untrained, on the same 85 held-out restarts with the live Operator.
3. Add solver hints at the starts where no correct move is ever sampled (28 of 132).
