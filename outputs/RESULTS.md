# RAILCAST prototype - results

Corridor: **NDLS - BCT, 1384 km, 202 stations, real working timetable**, 44 trains, 240 simulated operating days.

Data source: **real**. Station master, roster and working timetable from datameet/railways (8,697 stations, 5,208 trains, 417,080 scheduled calls, CC0 1.0 (datameet/railways)); weather is ERA5 reanalysis via Open-Meteo from 2024-01-01. Still invented: temporary speed restrictions, engineering blocks, run-time friction and incident rates, conflict outcomes (no live feed collected yet).

Temporal holdout - trained on days 0-167 (342,301 forecasts), calibrated on days 168-203 (74,732), scored on days 204-239 (73,268). A random split would leak and flatter every number.

## Headline

| Metric | Carry-forward (today) | RAILCAST | Change |
|---|---|---|---|
| MAE | 26.9 min | 10.9 min | **60% lower** |
| P90 absolute error | 77.0 min | 29.4 min | 62% lower |
| Uncertainty stated | none | 80% window, 78.4% achieved coverage | - |
| Mean window width | - | 28 min | - |

Conflict simulator alone, with the learned layer switched off, scores 10.7 min MAE overall - still well ahead of carry-forward, which is what makes it a usable fallback when the model is unavailable. It is worse than carry-forward at the next station, where repeating the current delay is hard to beat; the learned residual is what fixes that end of the range.

## By horizon

| Horizon | n | Carry-forward MAE | Simulator only | RAILCAST MAE | Improvement | Coverage | Window |
|---|---|---|---|---|---|---|---|
| Next station (<1 h) | 1,894 | 11.0 min | 5.8 min | 4.9 min | **55%** | 84.5% | 18 min |
| 1-3 h | 2,951 | 17.8 min | 6.4 min | 6.6 min | **63%** | 82.6% | 21 min |
| 3-8 h | 9,277 | 22.8 min | 8.9 min | 9.3 min | **59%** | 79.4% | 26 min |
| 8-24 h | 38,372 | 31.4 min | 12.3 min | 12.5 min | **60%** | 77.7% | 32 min |
| 24 h + | 20,774 | 23.4 min | 9.6 min | 9.8 min | **58%** | 78.0% | 25 min |

## Decision surfaces

- **Costed precedence.** One real conflict from day 232, re-run 24 times each way on identical days. Reversing the call saves 12925-Slip 263 minutes, costs 12925 +233 and the rest of the corridor +732 - +1228 network-minutes on net (standard error 156).
- **Connection risk.** Stated make-probability against realised outcome on held-out days: reliability error 0.006 over 40,000 forecasts.
- **Serving.** 0.46 ms per station forecast, 0.32 s for a full corridor re-forecast of 688 station predictions, single process, no GPU. Linear extrapolation to 13,000 trains: 93 s in one process, about 6 s sharded across 16 zonal graphs - inside the 30 s budget.

## The corridor the model learns on

Simulated punctuality: median 8 min late at destination, P90 135 min, 56% arriving within 15 minutes.

## Honesty note for the deck

**The corridor, the roster, the timetable and the weather are real; how delay arises is still modelled.** Run-time friction, incidents, conflict outcomes, speed restrictions and engineering blocks are all generated, so every number above is scored against simulated arrivals. The simulator's parameters were set by hand until punctuality looked plausible - they are not yet fitted to observed running data.

What this does establish: the forecaster never sees the noise that generates the delay, the baseline is computed on exactly the same rows, and the split is temporal. That is evidence the method works on a problem of the right shape - not evidence of field performance. Label every chart that way.
