# RAILCAST prototype - results

Corridor: **New Delhi - Mumbai Central (1386 km, 35 stations)**, 32 trains, 450 simulated operating days.

Temporal holdout - trained on days 0-314 (921,524 forecasts), calibrated on days 315-381 (195,616), scored on days 382-449 (197,472). A random split would leak and flatter every number.

## Headline

| Metric | Carry-forward (today) | RAILCAST | Change |
|---|---|---|---|
| MAE | 19.4 min | 8.5 min | **56% lower** |
| P90 absolute error | 54.7 min | 22.2 min | 59% lower |
| Uncertainty stated | none | 80% window, 81.3% achieved coverage | - |
| Mean window width | - | 27 min | - |

Conflict simulator alone, with the learned layer switched off, scores 11.7 min MAE overall - still well ahead of carry-forward, which is what makes it a usable fallback when the model is unavailable. It is worse than carry-forward at the next station, where repeating the current delay is hard to beat; the learned residual is what fixes that end of the range.

## By horizon

| Horizon | n | Carry-forward MAE | Simulator only | RAILCAST MAE | Improvement | Coverage | Window |
|---|---|---|---|---|---|---|---|
| Next station (<1 h) | 7,991 | 7.2 min | 19.7 min | 4.9 min | **32%** | 81.2% | 17 min |
| 1-3 h | 13,638 | 10.0 min | 14.0 min | 5.8 min | **42%** | 81.0% | 19 min |
| 3-8 h | 39,564 | 14.4 min | 10.7 min | 7.4 min | **49%** | 81.7% | 24 min |
| 8-24 h | 113,026 | 21.7 min | 11.1 min | 9.1 min | **58%** | 80.7% | 28 min |
| 24 h + | 23,253 | 26.5 min | 12.1 min | 10.6 min | **60%** | 84.0% | 35 min |

## Decision surfaces

- **Costed precedence.** One real conflict from day 390, re-run 24 times each way on identical days. Reversing the call saves 12915 32 minutes, costs 12953 +49 and the rest of the corridor +9 - +26 network-minutes on net (standard error 8).
- **Connection risk.** Stated make-probability against realised outcome on held-out days: reliability error 0.012 over 40,000 forecasts.
- **Serving.** 0.22 ms per station forecast, 0.16 s for a full corridor re-forecast of 719 station predictions, single process, no GPU. Linear extrapolation to 13,000 trains: 65 s in one process, about 4 s sharded across 16 zonal graphs - inside the 30 s budget.

## The corridor the model learns on

Simulated punctuality: median 16 min late at destination, P90 71 min, 49% arriving within 15 minutes.

## Honesty note for the deck

**Every number above comes from a simulator. No Indian Railways data was used, downloaded or consulted.** Station names, km posts and train numbers are approximate; timetables, running times, delays, conflicts and weather are all generated. The simulator's parameters were set by hand until punctuality looked plausible - they were not fitted to published punctuality returns, which is step 2 of the roadmap.

What this does establish: the forecaster never sees the noise that generates the delay, the baseline is computed on exactly the same rows, and the split is temporal. That is evidence the method works on a problem of the right shape - not evidence of field performance. Label every chart that way.
