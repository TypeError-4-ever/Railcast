# Railcast — Product Requirements

**Status:** draft v0.1 · 25 Oct 2026 · working name, not final

---

## 1. In one line

For people who have to travel in the next 48 hours: find a way from A to B that can
actually be booked — including connections through stations **off** the direct
line — and show the real chance of making the connection.

## 2. The problem

- Someone has a sudden plan: they need to travel tonight or tomorrow. Every direct
  train shows WL or REGRET, and tatkal sells out in minutes.
- Existing apps mostly look for alternatives **on** the direct line: the same train
  boarded a station earlier, or a change at a station the route already passes
  through. (To be verified — see §5.)
- When apps do suggest a connection, they show the **layover time**, not the chance
  of making it. Indian trains run late often, and two separate tickets mean a
  missed connection is the passenger's problem.
- So people give up, take an expensive bus, or gamble on a tight connection blind.

## 3. Who it is for

**Primary: sudden-plan travellers — journey starts within 0–48 hours.**

| Who | What they need |
|---|---|
| Student going home at short notice | Any confirmed way home; will pay a bit more, but fare still matters |
| Professional on a sudden work trip | Fastest confirmed route; fare matters less |
| Family emergency | Get there, on any confirmable route, now |

**Not for v1:** people booking weeks ahead. They mostly get confirmed tickets
anyway, and waitlist-prediction apps already serve them.

This focus also matches the model: Railcast forecasts up to ~30 hours ahead, which
is exactly the sudden-plan window.

## 4. How we rank options

Priority order:

1. **Chance you actually reach B**
   = P(seat on leg 1) × P(seat on leg 2) × P(making the connection)
   A confirmed seat on leg 2 is worthless if you miss it — so connection risk is
   part of priority 1, not an afterthought.
2. **Predicted real travel time** — from the delay model, not the timetable. A
   train that is "18 h" on paper but usually runs 4 h late is a 22 h train.
3. **Fare.**

**Group, don't strictly sort.** A strict sort makes a confirmed 38 h trip with two
changes beat an 18 h trip at 95%. Instead:

| Group | Chance of reaching B |
|---|---|
| Safe | 90% or more |
| Likely | 70–90% |
| Risky | below 70% |

Inside each group, sort by predicted time, then fare. Show **three labelled
picks** — Safest, Fastest, Cheapest — instead of one "best".

**Seat probability per leg, for the 0–48 h window:**

| Leg status | How we treat it |
|---|---|
| CNF available now | High, minus the chance it sells before the user books |
| RAC | Travel assured (you can board); flagged as a shared berth |
| WL | Low — an e-ticket still waitlisted when the chart is prepared is cancelled automatically, and under 48 h there is little time for it to clear |
| Booking window not open yet | Predicted by the seat-likelihood model (§8, F8) |

## 5. What already exists, and our gap

| Product | What it does |
|---|---|
| ConfirmTkt Alternates (owned by ixigo) | Unused quota on the same train from other stations; if that fails, combines two trains or a train and a bus. Says it accounts for delays |
| ixigo Same Train Alternates / Train Jugaad | Board earlier or get off later on the same train to find a confirmed seat |
| Trainman, RailYatri, IRCTC and others | Waitlist prediction, PNR status, live running status |

**Our gap (all unverified until the Phase 0 test):**

- **G1** — change stations **off** the direct line, through hubs on other corridors.
  Trains on another corridor draw from a different crowd, so they are more likely
  to have seats when the direct line is full.
- **G2** — a calibrated probability of making the connection, from simulating how
  the trains will actually run.
- **G3** — a **booking plan**: legs open for booking at different times; tell the
  user which leg to book first and when.
- **G4** — travel-day rerouting when leg 1 runs late.

**Honest note:** G1 and G3 are features; a big company can copy them in weeks. The
moat is **G2 plus the real running data behind it**.

## 6. Goals and non-goals

**Goals for v1**

- For a sudden-plan trip where direct trains are full, return at least one Safe
  option as often as possible.
- Every option shows its chance of reaching B, predicted arrival, fare, and how much
  extra time and distance it adds compared with the direct route.
- Tell the user exactly when and in what order to book.

**Non-goals**

- **We do not book tickets.** We link out to IRCTC.
- **We never say "guaranteed".** Availability changes by the minute. We say "best
  options that look bookable right now".
- **No LLM for route finding.** Routing is a solved graph problem; an LLM would be
  slower and wrong. An LLM may parse free-text searches later.
- **No advance-trip planning in v1.**
- **No email for urgent alerts.** Email is too slow for tatkal. Use Telegram or push.

## 7. User flow (v1)

1. **Search** — From, To, when (tonight / tomorrow / date), class, max changes
   (default 1, allow 2).
2. **Results** — three picks, each showing: legs and change station, layover,
   chance of making the connection, chance of reaching B, predicted arrival
   window, fare, extra time and km vs direct.
3. **Booking plan** — the order to book the legs, when each booking window opens,
   which leg is scarcer (book that one first, so nobody ends up with leg 1 and no
   way onward).
4. **Alerts** — one minute before each booking window opens; "leg 2 is selling
   fast"; "chart prepared, current booking open".
5. **Travel day** — track leg 1 live. If the chance of making the connection drops
   below a threshold, alert with plan B from the change station: the next train,
   or a bus.
6. **After a missed connection** — remind the user that IRCTC has a PNR-linking
   feature for refunds on missed connecting journeys (check the current rule
   before relying on it).

## 8. Features

### P0 — MVP

| ID | Feature | Notes |
|---|---|---|
| F1 | Route engine | Timetable graph search (Connection Scan Algorithm or RAPTOR). 0–2 changes. Change stations anywhere, not only on the direct line. Detour cap (start at ≤1.4× direct distance and ≤1.5× fastest time, then tune). Minimum layover. Same-city station groups with transfer times (NDLS↔NZM↔ANVT, CSMT↔LTT↔BCT, etc.) |
| F2 | Availability check | Via RailKit. Only for the top ~10 candidates after pruning. Cache results for a few minutes. Log every response |
| F3 | Connection probability | From the Railcast delay model. v1 uses delay history from collected live data; the simulator forecast takes over once trains are inside its ~30 h horizon |
| F4 | Ranking and three picks | As in §4 |
| F5 | Booking plan and alerts | Telegram bot first (free); push notifications later |
| F6 | Plan B at the change station | Next trains (and later buses) from C to B |

### P1

| ID | Feature | Notes |
|---|---|---|
| F7 | Travel-day rerouting | Live tracking of leg 1, alert when the connection is at risk |
| F8 | Seat-likelihood model | Predicts which legs will have seats, trained on logged availability, so we call the availability API far less. This is where "a model to cut cost" belongs |
| F9 | Waitlist/RAC chance per leg | For legs that are not CNF |

### Later

Train + bus combinations · natural-language search · advance planning mode ·
web app / Android app.

## 9. How it works

```
timetable ──► route engine (CSA / RAPTOR) ──► hundreds of candidate trips
                                                     │
              prune: detour cap · min layover · seat-likelihood (F8)
                                                     │
                                              top ~10 candidates
                                                     │
              RailKit availability (cached) ──► Railcast connection probability
                                                     │
                                           ranking ──► three picks
```

**Background jobs**

- **Live running logger** — `python -m railcast.feeds collect --trains ...`, on a
  schedule. Already in the repo; needs `RAILRADAR_API_KEY` and an always-on machine.
- **Re-forecast** — every ~15 minutes for trains departing in the next 48 h.
  Current benchmark: 0.32 s to re-forecast one corridor. The ~6 s figure for all
  13,000 trains (16 zonal shards) is an extrapolation, not a measurement.
- **Availability logger** — save every RailKit response; this becomes the training
  data for F8.

### What the repo already has, and what is missing

| Piece | Today | Needed |
|---|---|---|
| Timetable | datameet snapshot: 8,697 stations, 5,208 trains, 417,080 scheduled stops. An old snapshot | A fresher source; check for new and cancelled trains |
| Delay model | Railcast forecaster on one corridor (NDLS–BCT, 44 trains) | National coverage, fitted to real running data |
| Connection risk | `analysis.connection_analysis` — reliability error 0.006, but **scored on simulated delays** | The same check on real data |
| Live logging | `railcast/feeds/live.py`, `collect` command, RailRadar adapter | API key, a scheduler, the list of trains to watch |
| Fit to real data | `calibrate --from-observed` and `validate` commands | Collected data to run them on |
| Route engine | — | Build (F1) |
| Availability | — | RailKit adapter + logging (F2) |
| Alerts / UI | — | Telegram bot (F5) |

## 10. Booking windows (input to the booking plan)

- **Tatkal** opens one day before the journey date, counted from the day the train
  leaves its first station: **10:00 for AC classes, 11:00 for non-AC**.
- **Chart preparation** is about **8 hours before departure** (for trains leaving
  before 14:00, by 21:00 the previous night). After the chart, leftover berths go
  on **current booking** until shortly before departure.
- These rules change. Keep them in config, not hard-coded, and recheck them
  regularly.

## 11. How we measure success

**Model**

- Arrival-time error on **real** trains, compared with carry-forward and with the
  plain timetable.
- Calibration on real data: when we say 90%, about 90% of connections are made.

**Product**

- **Rescue rate** — of searches where direct trains are WL/REGRET, the share that
  return at least one Safe option.
- Share of users who book a suggested route.
- Share of suggested connections actually made.
- Alerts acted on.
- Weekly active users and repeat use.

**Cost**

- Availability API calls per search (target: ~20 or fewer) and cost per search.

## 12. Risks

| Risk | Why it matters | What we do |
|---|---|---|
| RailKit or RailRadar is unofficial, changes, or gets blocked | Product stops overnight | Read their terms and find out where their data comes from. Keep every provider behind an adapter (as `live.py` already does). Line up a second provider. Log everything |
| Model unproven on real delays | The core claim fails | Start logging now. Label every number "simulated" until it is checked on real data |
| Competitors copy off-line routing | Feature parity | The moat is real data plus accuracy. Move fast |
| No revenue without booking | No business | Free tool → users + data → licence or acquisition (§13) |
| At peak season, connecting legs are full too | Fewer rescues when it matters most | Off-line hubs draw from a different crowd; measure the rescue rate |
| API rate limits at 10:00 tatkal rush | Slow or failing at the worst moment | Prepare plans before the window opens; cache; test limits during tatkal hour |
| Seats sell between search and booking | Frustrated users | Show when availability was checked and "selling fast"; never promise |
| Stale timetable | Wrong routes | Fresher timetable source; let users report errors |

## 13. Business model and exit

Free for users in v1. What happens next depends on what we can show:

| What we bring | Likely outcome |
|---|---|
| A demo | Job or internship offer |
| Real accuracy numbers plus months of our own data | Licensing an API (paid per call), or the team gets hired for it |
| Thousands of active users | Acquisition — ConfirmTkt already had a big user base when ixigo bought it |

**Indian Railways is a buyer too.** The original SIH problem statement (dynamic ETA
forecasting) came from the Ministry of Railways; the forecaster is what they asked
for.

Possible revenue later: a paid alert pass for festival season; a B2B
connection-risk API priced per call.

## 14. Plan

### Phase 0 — this week (late Oct): prove the gap, start the data clock

- [ ] Get a RailRadar API key. Check the free tier's limits (watching N trains every
      15 minutes is N × 96 calls a day).
- [ ] Run `python -m railcast.feeds collect` on a schedule, on a machine that stays
      on (a cheap VPS, or a scheduled GitHub Action — a laptop sleeps). Pick the
      trains on 5–10 routes your target users actually travel.
- [ ] **Gap test:** 10 sudden-plan trips where direct trains are WL. Write down
      what ConfirmTkt and ixigo suggest for each. If they already show good
      off-line connections, rethink before building.
- [ ] RailKit: read its terms, find its data source, pricing and rate limits.
- [ ] Talk to 15–20 people who recently had to travel at short notice. The festival
      rush is happening now — use it.

**Done when:** the logger runs on its own, the gap test result is written down, and
RailKit is cleared (or an alternative found).

### Phase 1 — weeks 2–4: route engine (F1)

- [ ] Load the timetable into a graph; implement CSA or RAPTOR with 0–2 changes.
- [ ] Detour cap, minimum layover, same-city station groups.
- [ ] A command that prints the top itineraries for A → B on a date, by scheduled
      time.
- [ ] Check it on 20 real routes against what you would plan by hand.

### Phase 2 — weeks 4–6: availability and ranking (F2–F4)

- [ ] RailKit adapter, cache, availability logging.
- [ ] Connection probability v1 from logged delay history; simulator as fallback.
- [ ] Ranking groups, three picks, extra time and km vs direct.

### Phase 3 — weeks 6–8: Telegram bot beta (F5, F6)

- [ ] Search → three picks → booking plan → alerts → plan B.
- [ ] 50–100 beta users from friends and hostel.

**Done when:** people use it on real trips and the rescue rate is measured.

### Phase 4 — Dec to Feb (fog season): prove the model

- [ ] Run `calibrate --from-observed` and `validate` on the collected data.
- [ ] Retrain the residual model on real running data; compare with carry-forward;
      write down honest numbers.
- [ ] Extend beyond the NDLS–BCT corridor.
- [ ] Travel-day rerouting (F7).

Fog season has the worst delays — the hardest test, and the most convincing one.

### Phase 5 — Mar to May (summer rush): scale and pitch

- [ ] Seat-likelihood model (F8) from logged availability → fewer API calls.
- [ ] Grow users; measure repeat use.
- [ ] A one-page summary of real numbers → pitch to OTAs and Indian Railways.

## 15. Open questions

- Which 5–10 routes first?
- RailKit pricing and limits at scale?
- Where do we get a fresher timetable?
- Telegram, WhatsApp, or an app — what do target users actually use?
- Final product name.
