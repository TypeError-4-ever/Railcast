"""
The trip planner: from A to B in the next 48 hours, with up to two changes.

    network    the national timetable as a searchable graph
    search     itineraries with 0-2 changes, change stations anywhere
    delays     per-station delay distributions - prior, or learned from data
    risk       chance of making each connection and of reaching B
    seats      seat availability (RailKit) and what each status is worth
    rank       Safe / Likely / Risky, and the three picks
    booking    when each leg's booking window opens, and what to book first
    respond    the response a user sees, as text or JSON
    chart      one chart per response
"""
