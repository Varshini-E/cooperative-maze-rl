# What changed, and what helped

October 27, 2026. Manually maintained experiment notes. Full traces and earlier plans are archived locally.

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

## 4. What is still wrong with RL?

- **Format collapse:** original RL caused roughly 60% rejected Expert messages.
- **Stabilization:** lower learning rate + replay preserved formatting, but completion went **9/16 → 9/16** (V7) and **10/16 → 8/16** (V8).
- **Open problem:** updates work; reliable completion gains remain unproven. Sparse episode credit, partner errors and long responses receiving more loss weight are possible contributors, not established causes.
- **Current test:** thinking-enabled Expert-only RL, two easy tasks, groups of eight, `1e-6` learning rate, no replay; up to eight batches and fresh interleaved initial/final evaluation. This tests task fitting only.
