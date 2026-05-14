## Summary

Adds the **codex v2 legal** record: **bin = 3037 steps to 3.28 val_loss**, validated over **n=16 non-cherry-picked seeds (0..15)** with distinct `--seed N` per run.

This descends from the v12 optimizer stack but keeps the Track 3 architecture block compliant after the legal rebuild. The winning recipe layers:

1. Polar Express NS-5 + MuonEq row-normalized update.
2. Role-specific Muon LR multipliers: q/k `0.61875`, v `0.625`, attn.proj `0.6375`, mlp.fc `1.0125`, mlp.proj `0.9875` of base LR `0.045`.
3. Role-specific weight decay around `0.027..0.0315` instead of one body-wide value.
4. Muon lookahead from step 2450, interval 25, alpha 0.35, pull 0.15, with a 150-step smoothstep ramp.
5. Embed init `* 0.7`, AdamW betas `(0.8, 0.95)`, `eta_min=0.02`, and the `0.85 -> 0.95 -> 0.85` Muon schedule.

### Note on the bin

The 3025 checkpoint is a real crossing but not a statistically significant Track 3 bin under the fixed rule:

```text
3025: n=16, mean=3.27928750, margin=+0.00285000
3037: n=16, mean=3.27853000, margin=+0.00588000
```

The first controlled step that clears `(3.28 - mu) * sqrt(n) >= 0.004` is **3037**.

Log hygiene note: the seed 2..7 raw full logs had appended extra trials from an early wrapper that still parsed the trailing seed value as `num_trials`. The committed logfiles are truncated at the first complete controlled stream per seed; those are the streams used in the table and statistic below.

## Result

The submitted step count is **3037**. The result directory contains 16 full first-stream reproducibility logfiles for seeds 0 through 15.

At step 3037:

```text
n = 16
mean val loss = 3.27853000
std            = 0.00080602
(3.28 - mu) * sqrt(n) = 0.00588000
```

This exceeds the Track 3 README threshold of `0.004`. Equivalently, with `sigma=0.0013`, this is `z = 4.5231`, one-sided `p = 3.05e-06`, satisfying `p < 0.001`.

| Seed | 3000 val | 3025 val | 3037 val |
| -: | -: | -: | -: |
| 0 | 3.28137 | 3.27901 | 3.27825 |
| 1 | 3.28160 | 3.27926 | 3.27850 |
| 2 | 3.28228 | 3.27996 | 3.27921 |
| 3 | 3.28142 | 3.27907 | 3.27833 |
| 4 | 3.28145 | 3.27913 | 3.27836 |
| 5 | 3.28015 | 3.27784 | 3.27708 |
| 6 | 3.28333 | 3.28105 | 3.28031 |
| 7 | 3.28149 | 3.27918 | 3.27841 |
| 8 | 3.28163 | 3.27931 | 3.27860 |
| 9 | 3.28092 | 3.27861 | 3.27786 |
| 10 | 3.28267 | 3.28036 | 3.27959 |
| 11 | 3.28111 | 3.27878 | 3.27801 |
| 12 | 3.28104 | 3.27868 | 3.27791 |
| 13 | 3.28104 | 3.27869 | 3.27793 |
| 14 | 3.28272 | 3.28042 | 3.27965 |
| 15 | 3.28155 | 3.27925 | 3.27848 |
| **Mean** | **3.28161** | **3.27929** | **3.27853** |

## Comparison to earlier checkpoints

![loss curves](loss_curves.png)

## Stack contribution

![stack contribution](pruning.png)

Per-component contributions are from the `pruning-rerun` codex v2 legal sweep at step 3037. Lookahead and role LR/WD are the main v2-specific additions; MuonEq and the Muon schedule remain the largest inherited contributors. Raw numbers are in `pruning_data.json`.

## Files

- `records/track_3_optimization/results/20260515_codex_v2_legal_3037/README.md`
- `records/track_3_optimization/results/20260515_codex_v2_legal_3037/loss_curves.png`
- `records/track_3_optimization/results/20260515_codex_v2_legal_3037/pruning.png`
- `records/track_3_optimization/results/20260515_codex_v2_legal_3037/pruning_data.json`
- 16 full first-stream reproducibility logfiles, seeds 0..15

## Credits

- [@nilin PR #275 / Contra-Muon](https://github.com/KellerJordan/modded-nanogpt/pull/275): Contra-Muon shaping lineage.
- [Polar Express](https://arxiv.org/abs/2505.16932) NS-5 and [MuonEq](https://arxiv.org/abs/2603.28254) row-normalized Muon update.
- Codex v2 additions: compliant v12 rebuild, role-specific LR/WD, and Muon lookahead.
