# M8 Cross-Section Noise Scaling (Route A Experiment 5)

Daily-IC std vs cross-section size k, averaged over 21 runs (all Part D families x 3 seeds); subsampled ticker subsets per test day.

| k (stocks per day) | measured sigma(daily IC) | theory 1/sqrt(k-1) |
|---:|---:|---:|
| 3 | 0.709 | 0.707 |
| 4 | 0.580 | 0.577 |
| 5 | 0.501 | 0.500 |
| 6 | 0.446 | 0.447 |
| 7 | 0.404 | 0.408 |

- Measured sigma at k=7 is 0.404 vs theoretical floor 0.408 for zero-signal cross-sections — the benchmark noise is essentially the mathematical floor of 7-name daily correlations, not a model artifact.
- Standard error of a 246-day mean IC at k=7: 0.0258 (single run, before seed/selection variance).
- Universe size needed to resolve Delta IC = 0.02 at 95%: k ≈ 41 stocks; Delta IC = 0.01: k ≈ 158 stocks (holding 246 test days).

Implication: at k=7 the benchmark cannot distinguish models closer than roughly 0.05 IC even before training variance is considered; architectural effects of 0.01-0.02 are unresolvable by design.
