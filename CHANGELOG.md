# What changed, and what helped

Updated November 16, 2026. Manually maintained experiment notes. Full traces and earlier plans are archived locally.

## 1. Better communication

- **Explicit actions:** `ask` requests information; `instruct` requests an action. Only validated `move`/`toggle` calls change state. Receipts confirm execution; toggle means reverse, not “ensure ON.”
- **Reliable answer delivery:** V6 stopped discarding replies on mislabeled turns. V7 replaced the copied “observations” placeholder with factual examples. Both fixed demonstrated failures.
- **Usable context:** overview plus room crop; two recent exchanges and receipts for the Operator; current images only. Follow-ups became possible, but memory alone has no established completion gain.
- **Shared vocabulary:** V8 added canonical symbols and immediate failure for invalid Expert messages. Protocol compliance now matters to success; a completion gain was not established.

## 2. Thinking

Reasoning improved completion without training: initial Expert **11/24 → 17/24**, SFT **9/24 → 15/24**. This was the clearest inference improvement in our small development sample.

The 8,192-token cap removed sampled truncations; initial completion stayed **8/12**. We now use initial + thinking + 8,192.

## 3. Warm start

SFT used **307 reviewed Expert messages**. Development completion rose **6/16 → 9/16**; held-out completion was **20/48 → 22/48**, with mixed transfer.

With thinking, SFT scored **15/24 versus initial 17/24**. There is no consistent warm-start advantage. The old demonstrations were non-thinking; current RL omits them.

## 4. RL

- **Format collapse:** the original token-summed loss at LR 4.7e-4 caused ~60% rejected Expert messages.
- **Too small a step:** at LR 1e-6, format held but completion went **12/16 → 8/16**.
- **Length bias:** failed episodes are longer, so token-summed loss weighted them **2.2–6.8×** more than successes. An overfit test selected per-episode normalization at LR 1e-4.
- **Fitting, not generalizing:** a 4-task pilot improved **20/32 → 25/32**; on unseen tasks it scored **37 vs 36 of 48**, and a broader run **81 vs 82 of 120**.
- **Bottleneck:** a scripted correct Expert with the real Operator solved **15/16**; the Expert's planning limits success.
- **More optimization per batch:** three clipped PPO steps move success tokens ~9× more than one step, with ≤2% clipped.
- **Restart points:** training from recorded mid-episode states focused groups on hard decisions; held-out **42 vs 41 of 85**.
- **Systematic errors:** 95% of wrong choices repeat the same wrong door, and a manual hint gave only +4.9 points.
- **Denser signal:** solver progress credit with 10 updates on 4 decisions raised correct first messages from **7.8% to 17.2%**. Fine-tuning on 273 solver-checked turns raised held-out first-move accuracy from **36.8% to 38.2%** (CI −4.4 to +7.8). Neither met its bar.
- **Cheaper rollouts:** a provably lost episode now stops early (~19% fewer tokens), and the spending guard settles reservations to actual token cost (previously ~2×).

See [RL status](docs/rl-status.md).
