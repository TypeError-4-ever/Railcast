"""
The delay forecaster: conflict simulator, learned residual, calibrated window.

    corridor / trains / simulator / noise / simconfig   the simulated railway
    dataset / models                                    stages A-D
    evaluate / analysis                                 scoring against carry-forward
    calibrate / validate                                fitting to collected running data
    pipeline                                            the end-to-end run
    charts                                              the three charts that matter

It forecasts one corridor at a time from live train positions. The trip planner
(`railcast.planner`) uses its own per-station delay model for trains that have
not started yet, and will hand running trains to this stack.
"""
