# Forecaster results

Corridor: **NDLS - BCT, 1384 km, 202 stations, real working timetable**, 48 trains, 240 simulated operating days. Trained on days 0-167, calibrated on 168-203, scored on 204-239 (temporal holdout).

**Scored on simulated delays.** The corridor, timetable and weather are real; how delay arises is still modelled, so these numbers show the method works on a problem of the right shape - not field accuracy. They become real once the simulator is fitted to collected running data (`python -m railcast.feeds calibrate --from-observed`).

| Metric | Carry-forward (today) | RAILCAST |
|---|---|---|
| MAE | 28.0 min | 11.1 min (60% lower) |
| P90 absolute error | 80.8 min | 30.3 min |
| 80% arrival window | none | 79.1% coverage, 29 min wide |
| Connection probability | none | reliability error 0.008 over 40,000 forecasts |

## By horizon

| Horizon | n | Carry-forward | RAILCAST | Improvement | Coverage |
|---|---|---|---|---|---|
| Next station (<1 h) | 2,438 | 10.6 min | 4.8 min | 55% | 82.2% |
| 1-3 h | 4,026 | 16.2 min | 5.7 min | 65% | 81.6% |
| 3-8 h | 13,215 | 22.8 min | 8.8 min | 62% | 79.4% |
| 8-24 h | 55,306 | 33.4 min | 12.8 min | 62% | 78.9% |
| 24 h + | 29,628 | 23.3 min | 10.4 min | 55% | 78.8% |

Serving: 0.10 s to re-forecast the corridor (723 station forecasts), one process, no GPU.

Charts: `outputs/charts/` - accuracy by horizon, connection risk, one journey forecast.
