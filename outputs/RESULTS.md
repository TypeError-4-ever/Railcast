# RAILCAST prototype - results

Corridor: **NDLS - BCT, 1384 km, 202 stations, real working timetable**, 44 trains, 240 simulated operating days.

Data source: **real**. Station master, roster and working timetable from datameet/railways (8,697 stations, 5,208 trains, 417,080 scheduled calls, CC0 1.0 (datameet/railways)); weather is ERA5 reanalysis via Open-Meteo from 2024-01-01. Still invented: temporary speed restrictions, engineering blocks, run-time friction and incident rates, conflict outcomes (no live feed collected yet).

Temporal holdout - trained on days 0-167 (342,891 forecasts), calibrated on days 168-203 (74,825), scored on days 204-239 (73,431). A random split would leak and flatter every number.

## Headline

| Metric | Carry-forward (today) | RAILCAST | Change |
|---|---|---|---|
| MAE | 37.5 min | 13.6 min | **64% lower** |
| P90 absolute error | 113.1 min | 33.2 min | 71% lower |
| Uncertainty stated | none | 80% window, 78.8% achieved coverage | - |
| Mean window width | - | 38 min | - |

Conflict simulator alone, with the learned layer switched off, scores 13.5 min MAE overall - still well ahead of carry-forward, which is what makes it a usable fallback when the model is unavailable. It is worse than carry-forward at the next station, where repeating the current delay is hard to beat; the learned residual is what fixes that end of the range.

## By horizon

| Horizon | n | Carry-forward MAE | Simulator only | RAILCAST MAE | Improvement | Coverage | Window |
|---|---|---|---|---|---|---|---|
| Next station (<1 h) | 2,057 | 15.4 min | 6.7 min | 6.1 min | **60%** | 81.3% | 21 min |
| 1-3 h | 2,951 | 23.8 min | 7.6 min | 7.7 min | **68%** | 79.5% | 24 min |
| 3-8 h | 9,277 | 33.9 min | 11.5 min | 11.4 min | **66%** | 81.1% | 35 min |
| 8-24 h | 38,372 | 43.9 min | 15.9 min | 16.0 min | **64%** | 78.1% | 43 min |
| 24 h + | 20,774 | 31.6 min | 11.4 min | 11.5 min | **64%** | 78.5% | 32 min |

## Decision surfaces

- **Costed precedence.** One real conflict from day 232, re-run 24 times each way on identical days. Reversing the call saves 19024 2 minutes, costs 12288 +6 and the rest of the corridor -6 - -3 network-minutes on net (standard error 18).
- **Connection risk.** Stated make-probability against realised outcome on held-out days: reliability error 0.006 over 40,000 forecasts.
- **Serving.** 0.79 ms per station forecast, 0.54 s for a full corridor re-forecast of 689 station predictions, single process, no GPU. Linear extrapolation to 13,000 trains: 160 s in one process, about 10 s sharded across 16 zonal graphs - inside the 30 s budget.

## The corridor the model learns on

Simulated punctuality: median 22 min late at destination, P90 150 min, 46% arriving within 15 minutes.

## Honesty note for the deck

**Every number above comes from a simulator. No Indian Railways data was used, downloaded or consulted.** Station names, km posts and train numbers are approximate; timetables, running times, delays, conflicts and weather are all generated. The simulator's parameters were set by hand until punctuality looked plausible - they were not fitted to published punctuality returns, which is step 2 of the roadmap.

What this does establish: the forecaster never sees the noise that generates the delay, the baseline is computed on exactly the same rows, and the split is temporal. That is evidence the method works on a problem of the right shape - not evidence of field performance. Label every chart that way.
