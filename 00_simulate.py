# Databricks notebook source
# MAGIC %md
# MAGIC # 00: simulate the fleet data
# MAGIC
# MAGIC - This notebook makes the data that `01_replacement_analysis` and `02_detailed_checks` read. Run it first.
# MAGIC - A fleet of 250 power units, each with one replaceable component. Every unit is run forward from its start to 2026-09-30 and its components wear out with use.
# MAGIC - There is no company data here. The settings are round numbers.
# MAGIC - The random numbers start from a fixed seed, so every run gives the same data.
# MAGIC - It writes six tables: the units, the health factor of every period, a sample of raw data rows, the service reports, the actual replacements, and the actual remaining life of every unit. The last two are used only to check results.

# COMMAND ----------

import numpy as np
import pandas as pd

CATALOG = "workspace"
SCHEMA  = "pge_demo"

SEED    = 20261110
N_UNITS = 250
END     = pd.Timestamp("2026-09-30")     # every record ends here

HOURS_MEAN         = 4000      # operating hours on a unit: normal distribution
HOURS_SD           = 1500
HOURS_MIN          = 1000      # held between these two values
HOURS_MAX          = 8000
HOURS_PER_WEEK     = 40.0      # operating hours per calendar week
WEEK_SPREAD        = 0.15      # unit to unit
PERIOD_HOURS       = 20        # operating hours in one period
READINGS_PER_PERIOD = 120      # readings kept in one period
MINIMUM_LOAD_KW    = 20.0      # below this the unit is starting, stopping or idle
RAW_PER_SECOND     = 2.0       # raw readings per second
RAW_HOURS          = 6         # hours of raw data per unit
REPLACEMENT_LEVEL  = 0.50      # loss of performance that calls for replacement
HEALTH_WHEN_NEW    = 1.00      # one unit type, so one value
NOMINAL_POWER_KW   = 50.0

PERIOD_NOISE = 0.036           # scatter from one period to the next
DRIFT        = 0.014           # a slower wander on top

# life model: operating hours to the replacement level, by how the part was fitted.
# a part fitted on its own after a failure (A) goes into a unit whose other parts are worn, and
# lasts shorter. a part fitted along with other parts at scheduled work (B) lasts longer.
LIFE     = {"new": 5800, "A": 5000, "B": 6000}
LIFE_SD  = 800
LIFE_MIN = 3500
LIFE_MAX = 9000

VISIT_MIN_HOURS = 1000         # operating hours between scheduled visits, between these two
VISIT_MAX_HOURS = 1800
CHANGE_CHANCE   = 0.12         # chance that the part is changed at a scheduled visit
MINIMUM_LOSS    = 0.20         # but never before it has lost this much
FAILURE_LINE    = 0.40         # at or above this loss, an inspection finds a failure (A)
DAYS_OUT_MIN    = 4            # days out of service for a change at a scheduled visit
DAYS_OUT_MAX    = 12

rng = np.random.default_rng(SEED)
_ = spark.sql("CREATE SCHEMA IF NOT EXISTS " + CATALOG + "." + SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The wear curve
# MAGIC - A part loses performance slowly for about its first 2,000 operating hours, then faster. The point where it speeds up is called the knee.
# MAGIC - Each part gets its own knee, its own slow rate, and its own life. The fast rate is then calculated so that the loss reaches the replacement level exactly at the end of that life.
# MAGIC - The change from the slow rate to the fast rate is smooth, over a few hundred hours around the knee.

# COMMAND ----------

def loss_of(hours, part):
    # fraction of performance lost after this many operating hours on the part.
    # the rate moves smoothly from the slow rate to the fast rate around the knee.
    hours = np.asarray(hours, dtype=float)
    m1    = part["slow_rate"]
    m2    = part["fast_rate"]
    knee  = part["knee"]
    tau   = part["knee_width"]
    return (m1 * hours + (m2 - m1) * tau
            * (np.logaddexp(0.0, (hours - knee) / tau) - np.logaddexp(0.0, -knee / tau)))

def new_part(life):
    # the knee, its width and the slow rate are drawn at random.
    # the fast rate follows from the life.
    knee = rng.uniform(1800, 2400)
    tau  = rng.uniform(150, 300)
    m1   = rng.uniform(0.08, 0.14) * REPLACEMENT_LEVEL / knee
    part = {"slow_rate": m1, "fast_rate": m1, "knee": knee, "knee_width": tau, "life": life}
    a    = tau * (np.logaddexp(0.0, (life - knee) / tau) - np.logaddexp(0.0, -knee / tau))
    part["fast_rate"] = m1 + (REPLACEMENT_LEVEL - m1 * life) / a
    return part

HOURS_GRID = np.arange(0.0, 14000.0, 5.0)

def hours_at_loss(part, loss):
    # operating hours on the part at which it reaches a given loss
    return float(np.interp(loss, loss_of(HOURS_GRID, part), HOURS_GRID))

def draw_life(fitted_by):
    life = rng.normal(LIFE[fitted_by], LIFE_SD)
    return float(np.clip(life, LIFE_MIN, LIFE_MAX))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Units, and what happened to each part
# MAGIC - A part comes out in one of two ways.
# MAGIC   - It reaches the replacement level. Each site judges that point a little differently, so the loss at replacement scatters around 0.50.
# MAGIC   - Or the unit is in the shop for other scheduled work and the part is changed while it is open. That happens with a fixed chance at each visit, and only to a part that has already lost at least 20%.
# MAGIC - A change during scheduled work takes the unit out of service for 4 to 12 days.
# MAGIC - Class: what an inspection finds. Failure (A) if the part had lost 40% or more, non-failure (B) below that.
# MAGIC - The reason for the visit is a separate thing. It is what goes in the service report.
# MAGIC - The part fitted next gets its life from the class of the replacement that fitted it.

# COMMAND ----------

SITES = []
for number in range(1, 19):
    SITES.append("S-" + str(number).zfill(2))
CUSTOMERS = ["C-1", "C-2", "C-3", "C-4", "C-5"]

# sites are uneven: a few large ones and many small ones
UNITS_AT_SITE = [40, 34, 29, 24, 20, 17, 15, 12, 10, 9, 8, 7, 6, 5, 5, 4, 3, 2]
SITE_OF_UNIT  = []
for site, count in zip(SITES, UNITS_AT_SITE):
    for k in range(count):
        SITE_OF_UNIT.append(site)

units = []
for i in range(N_UNITS):
    unit_hours     = float(np.clip(rng.normal(HOURS_MEAN, HOURS_SD), HOURS_MIN, HOURS_MAX))
    hours_per_week = HOURS_PER_WEEK * rng.uniform(1 - WEEK_SPREAD, 1 + WEEK_SPREAD)

    # scheduled visits, and the ones at which the part can be changed
    first_visit = rng.uniform(VISIT_MIN_HOURS, VISIT_MAX_HOURS)
    visit_gap   = rng.uniform(VISIT_MIN_HOURS, VISIT_MAX_HOURS)
    visits      = np.arange(first_visit, HOURS_MAX, visit_gap)
    opened      = visits[rng.random(len(visits)) < CHANGE_CHANCE]

    # days out of service for a replacement after a failure.
    # short if it was done in a planned shutdown
    if rng.random() < 0.7:
        failure_days_out = rng.uniform(4, 12)
    else:
        failure_days_out = rng.uniform(0.1, 0.8)

    parts      = []
    part_start = 0.0          # unit hours when the part went in
    fitted_by  = "new"
    while True:
        part = new_part(draw_life(fitted_by))
        part["start"] = part_start
        if len(parts) == 0:
            part["loss_when_fitted"] = 0.0
        else:
            part["loss_when_fitted"] = rng.uniform(0.0, 0.12) * REPLACEMENT_LEVEL

        # the level is read off a noisy signal, so the call to replace scatters around it
        call = rng.normal(REPLACEMENT_LEVEL, 0.05 * REPLACEMENT_LEVEL)
        call = float(np.clip(call, 0.85 * REPLACEMENT_LEVEL, 1.15 * REPLACEMENT_LEVEL))
        fails_at      = hours_at_loss(part, call - part["loss_when_fitted"])
        can_change_at = hours_at_loss(part, MINIMUM_LOSS - part["loss_when_fitted"])

        # the first scheduled visit after the part has lost 20%
        later_visits = opened[opened >= part_start + can_change_at]
        if len(later_visits) > 0:
            shop_at = float(later_visits[0] - part_start)
        else:
            shop_at = np.inf

        replaced_at = min(fails_at, shop_at)
        if replaced_at >= unit_hours - part_start:
            # still fitted at the end of the record
            part["ends"]      = np.nan
            part["class"]     = None
            part["scheduled"] = None
            parts.append(part)
            break

        part["ends"]     = replaced_at
        part["loss_at"]  = part["loss_when_fitted"] + float(loss_of(replaced_at, part))
        if part["loss_at"] >= FAILURE_LINE:
            part["class"] = "A"
        else:
            part["class"] = "B"
        part["scheduled"] = shop_at <= fails_at          # why the unit was in the shop
        if part["scheduled"]:
            part["days_out"] = rng.uniform(DAYS_OUT_MIN, DAYS_OUT_MAX)
        else:
            part["days_out"] = failure_days_out
        parts.append(part)
        part_start = part_start + replaced_at
        fitted_by  = part["class"]

    health_when_new = HEALTH_WHEN_NEW * rng.uniform(0.97, 1.03)      # unit to unit spread
    nominal_power   = NOMINAL_POWER_KW * rng.uniform(0.96, 1.04)
    units.append({"unit_id": "U-" + str(100 + i),
                  "site": SITE_OF_UNIT[i],
                  "customer": CUSTOMERS[i % 5],
                  "hours": unit_hours,
                  "hours_per_week": hours_per_week,
                  "parts": parts,
                  "health_when_new": health_when_new,
                  "nominal_power": nominal_power})

def unit_loss(unit, hours):
    # loss of the unit at the given unit hours: the loss of whichever part was fitted then
    hours = np.asarray(hours, dtype=float)
    loss  = np.zeros_like(hours)
    for part in unit["parts"]:
        if np.isnan(part["ends"]):
            part_end = np.inf
        else:
            part_end = part["start"] + part["ends"]
        fitted = (hours >= part["start"]) & (hours < part_end)
        loss[fitted] = part["loss_when_fitted"] + loss_of(hours[fitted] - part["start"], part)
    return loss

# a count of what happened
replaced_parts = []
for unit in units:
    for part in unit["parts"]:
        if not np.isnan(part["ends"]):
            replaced_parts.append(part)
failures     = 0
non_failures = 0
scheduled    = 0
for part in replaced_parts:
    if part["class"] == "A":
        failures = failures + 1
    else:
        non_failures = non_failures + 1
    if part["scheduled"]:
        scheduled = scheduled + 1
print(pd.DataFrame({"count": [len(replaced_parts), failures, non_failures, scheduled,
                              len(replaced_parts) - scheduled]},
                   index=["replacements", "failure (A)", "non-failure (B)",
                          "unit was in the shop for scheduled work",
                          "unit was in the shop for a performance problem"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Period table
# MAGIC - One row per unit per 20 operating hours.
# MAGIC - Inside each period, readings are generated: load, power, differential pressure and temperature. The health factor is differential pressure divided by power on every reading, and the period keeps the average over the readings above the minimum load.
# MAGIC - Scatter: every period is moved up or down a little, and a slower drift is added on top. So the wear trend has to be found; it cannot be read off directly.
# MAGIC - About 3% of periods are missing, as in field data.
# MAGIC - Calendar time: a unit runs about 40 operating hours a week, with some idle days. A replacement takes it out of service for some days.

# COMMAND ----------

rows     = []
calendar = {}
for unit in units:
    n_periods       = int(unit["hours"] // PERIOD_HOURS)
    days_per_period = PERIOD_HOURS / (unit["hours_per_week"] / 7.0)

    # idle days before some periods, and days out of service at each replacement
    is_idle   = rng.random(n_periods + 1) < 0.015
    idle_days = rng.uniform(1.5, 5.0, n_periods + 1)
    gaps      = np.where(is_idle, idle_days, 0.0)
    for part in unit["parts"]:
        if not np.isnan(part["ends"]):
            j = int((part["start"] + part["ends"]) // PERIOD_HOURS) + 1
            if j <= n_periods:
                gaps[j] += part["days_out"]

    # calendar day at the start of every period, counted back from the end of the record
    day = np.cumsum(gaps) + np.arange(n_periods + 1) * days_per_period
    day = day - (day[n_periods - 1] + days_per_period)
    calendar[unit["unit_id"]] = (day, days_per_period)

    # scatter of each period, and a slower drift
    scatter = rng.normal(0, PERIOD_NOISE, n_periods + 1)
    drift   = np.zeros(n_periods + 1)
    for j in range(1, n_periods + 1):
        drift[j] = 0.92 * drift[j - 1] + rng.normal(0, DRIFT)

    for w in range(n_periods):
        if rng.random() < 0.03:                    # no data for this period
            continue
        if rng.random() > 0.08:
            n = READINGS_PER_PERIOD
        else:
            n = int(rng.integers(30, 90))          # a period with fewer readings
        hours = w * PERIOD_HOURS + (np.arange(n) + 0.5) * PERIOD_HOURS / n

        load   = np.clip(rng.normal(0.85, 0.07, n), 0.45, 1.05)
        power  = unit["nominal_power"] * load
        health = unit["health_when_new"] * (1.0 - unit_loss(unit, hours)) * (1.0 + drift[w] + scatter[w])
        noise_1 = rng.normal(0, 0.04, n)
        noise_2 = rng.normal(0, 0.3, n)
        dp      = np.clip(health * power * (1 + noise_1) + 0.6 + noise_2, 0, None)    # 0.6 is a sensor offset
        temp    = rng.normal(42.0, 2.5, n)

        health_factor = dp / power                       # on every reading
        above_minimum = power >= MINIMUM_LOAD_KW

        rows.append((unit["unit_id"], w, (w + 1) * PERIOD_HOURS,
                     (END + pd.Timedelta(days=float(day[w]))).round("s"),
                     (END + pd.Timedelta(days=float(day[w] + days_per_period))).round("s"),
                     n, float(power.mean()), float(dp.mean()), float(dp.std()), float(temp.mean()),
                     float(health_factor[above_minimum].mean())))

period_table = pd.DataFrame(rows, columns=[
    "unit_id", "period_id", "op_hours_cum", "ts_start", "ts_end",
    "n_samples", "power_kw_mean", "dp_mean", "dp_sd", "temp_c_mean", "health_factor"])
print(pd.DataFrame({"value": [len(period_table), str(period_table.ts_start.min().date()),
                              str(period_table.ts_end.max().date())]},
                   index=["periods", "record starts", "record ends"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## The actual replacements, and what the field wrote down
# MAGIC - First every replacement that happened, with the class an inspection would find.
# MAGIC - Then the service reports, with the faults of real reports:
# MAGIC   - about 1 in 10 replacements has no report
# MAGIC   - about 1 in 5 reports carries the wrong date
# MAGIC   - the reason is left blank in about a quarter, and is the wrong one in some others
# MAGIC   - the reason is the reason for the visit, not the condition of the part
# MAGIC   - the part is inspected after most replacements, not all

# COMMAND ----------

def date_at(unit_id, hours):
    # calendar date at which the unit reached the given operating hours
    day, days_per_period = calendar[unit_id]
    w = min(int(hours // PERIOD_HOURS), len(day) - 1)
    days = float(day[w] + days_per_period * (hours % PERIOD_HOURS) / PERIOD_HOURS)
    return END + pd.Timedelta(days=days)

rows = []
for unit in units:
    for part in unit["parts"]:
        if np.isnan(part["ends"]):
            continue
        hours = part["start"] + part["ends"]
        rows.append({"unit_id": unit["unit_id"],
                     "op_hours": round(hours, 1),
                     "ts": date_at(unit["unit_id"], hours),
                     "actual_class": part["class"],
                     "loss_at": round(part["loss_at"], 4),
                     "scheduled_visit": bool(part["scheduled"]),
                     "days_out": round(part["days_out"], 1),
                     "hours_per_week": unit["hours_per_week"]})
events = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

print(pd.crosstab(events.actual_class, np.where(events.scheduled_visit, "scheduled work", "performance")).to_string())
print()
print(events.groupby("actual_class").loss_at.describe()[["count", "min", "50%", "max"]].round(3).to_string())

# COMMAND ----------

PERFORMANCE_REASONS = ["low output", "performance below spec", "unit derated", "customer complaint"]
SCHEDULED_REASONS   = ["scheduled service", "preventive replacement", "planned maintenance",
                       "service interval"]

rows = []
for i, event in events.iterrows():
    if rng.random() < 0.10:                        # never written up
        continue

    if rng.random() < 0.18:                        # the date was recorded wrongly
        days_off = float(rng.uniform(-40, 60))
    else:
        days_off = 0.0

    if event.scheduled_visit:
        right_reasons = SCHEDULED_REASONS
        wrong_reasons = PERFORMANCE_REASONS
    else:
        right_reasons = PERFORMANCE_REASONS
        wrong_reasons = SCHEDULED_REASONS
    r = rng.random()
    if r < 0.25:
        reason = None                              # left blank
    elif r < 0.40:
        reason = str(rng.choice(wrong_reasons))    # recorded wrongly
    else:
        reason = str(rng.choice(right_reasons))

    if rng.random() < 0.85:                        # the part was inspected
        inspected_class = event.actual_class
    else:
        inspected_class = None

    rows.append({"unit_id": event.unit_id,
                 "service_date": (event.ts + pd.Timedelta(days=days_off)).round("s"),
                 "op_hours_at_service": round(float(event.op_hours)
                                              + days_off * float(event.hours_per_week) / 7.0, 1),
                 "reason": reason,
                 "inspected_class": inspected_class})
reports = pd.DataFrame(rows).sort_values("service_date").reset_index(drop=True)

print(pd.DataFrame({"count": [len(events), len(reports), int(reports.inspected_class.notna().sum()),
                              int(reports.reason.isna().sum())]},
                   index=["replacements", "service reports", "reports with an inspection",
                          "reports with the reason left blank"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## What actually happened
# MAGIC - These two tables exist only because the data is simulated. The method never reads them. They are used to check its results.
# MAGIC - Actual replacements: every replacement with its actual class.
# MAGIC - Actual remaining life: for the part now fitted in every unit, the operating hours until it reaches the replacement level, and the date that falls on if the unit keeps running at its usual hours per week. The wear curve of every part is known beyond the last data point, so this can be calculated.

# COMMAND ----------

actual_replacements = events[["unit_id", "op_hours", "ts", "actual_class", "loss_at",
                              "scheduled_visit", "days_out"]].copy()

rows = []
for unit in units:
    part          = unit["parts"][-1]                    # the part now fitted
    hours_on_part = unit["hours"] - part["start"]
    hours_to_level = hours_at_loss(part, REPLACEMENT_LEVEL - part["loss_when_fitted"])
    remaining     = hours_to_level - hours_on_part

    # the class of the replacement that fitted this part, if there was one
    replaced = []
    for earlier_part in unit["parts"]:
        if not np.isnan(earlier_part["ends"]):
            replaced.append(earlier_part)
    if len(replaced) > 0:
        fitted_at_class = replaced[-1]["class"]
    else:
        fitted_at_class = None

    rows.append({"unit_id": unit["unit_id"],
                 "fitted_at_class": fitted_at_class,
                 "n_replacements": len(replaced),
                 "hours_on_unit": round(unit["hours"], 1),
                 "hours_on_part": round(hours_on_part, 1),
                 "knee": round(part["knee"], 1),
                 "hours_remaining": round(remaining, 1),
                 "loss_now": round(float(unit_loss(unit, [unit["hours"]])[0]), 4),
                 "due_date": (END + pd.Timedelta(days=remaining / (unit["hours_per_week"] / 7.0))).round("s")})
actual_life = pd.DataFrame(rows)

print(pd.DataFrame({"count": [int((actual_life.hours_remaining <= 0).sum())]},
                   index=["units still running past the replacement level"]).to_string())
print()
print(actual_life.hours_remaining.describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw table
# MAGIC - Six hours of raw readings, two per second, for five units: two whose part was fitted at a failure replacement, one at a non-failure replacement, and two never replaced.
# MAGIC - Each unit runs, stops for a while, and runs again. So running time and idle time can be told apart.
# MAGIC - Power is not recorded. It has to be calculated from current and voltage.

# COMMAND ----------

units_by_id = {}
for unit in units:
    units_by_id[unit["unit_id"]] = unit

never_replaced = actual_life[actual_life.fitted_at_class.isna()].sort_values("loss_now")
chosen = (list(actual_life[actual_life.fitted_at_class == "A"].unit_id.head(2))
          + list(actual_life[actual_life.fitted_at_class == "B"].unit_id.head(1))
          + [never_replaced.unit_id.iloc[-1], never_replaced.unit_id.iloc[len(never_replaced) // 2]])
print(actual_life[actual_life.unit_id.isin(chosen)][["unit_id", "fitted_at_class", "hours_on_part",
                                                    "loss_now"]].to_string(index=False))

n       = int(RAW_HOURS * 3600 * RAW_PER_SECOND)
seconds = np.arange(n) / RAW_PER_SECOND
ramp    = int(120 * RAW_PER_SECOND)                      # a 2-minute ramp at each stop and start

tables = []
for unit_id in chosen:
    unit     = units_by_id[unit_id]
    at_hours = 0.9 * unit["hours"]
    health   = unit["health_when_new"] * (1.0 - float(unit_loss(unit, [at_hours])[0]))
    start    = date_at(unit_id, at_hours).round("h")

    # the unit stops and starts again at its own times
    stop_at    = rng.uniform(1.6, 2.9) * 3600
    restart_at = stop_at + rng.uniform(0.6, 1.5) * 3600
    running    = ((seconds < stop_at) | (seconds > restart_at)).astype(float)
    # a moving average turns each stop and start into a ramp
    padded     = np.r_[np.full(ramp, running[0]), running, np.full(ramp, running[-1])]
    running    = np.convolve(padded, np.ones(ramp) / ramp, mode="same")[ramp:-ramp]

    # its own load: a repeating swing, a slow wander and noise
    swing_size   = rng.uniform(0.02, 0.05)
    swing_period = rng.uniform(1100, 2800)
    swing_shift  = rng.uniform(0, 2 * np.pi)
    swing        = swing_size * np.sin(2 * np.pi * seconds / swing_period + swing_shift)
    wander       = np.convolve(rng.normal(0, 1, n), np.ones(900) / 900, "same") * 30 * rng.uniform(0.01, 0.03)
    load         = rng.uniform(0.76, 0.92) + swing + wander + rng.normal(0, 0.02, n)
    power        = unit["nominal_power"] * np.clip(load, 0.4, 1.1) * running
    voltage      = np.where(running > 0.5, 540 - 21 * (power / unit["nominal_power"]), 620) + rng.normal(0, 1.5, n)
    current      = np.where(running > 0.5, power * 1000 / voltage, rng.normal(0, 0.3, n))
    temp         = 25 + 20 * running + rng.normal(0, 0.4, n)
    noise_1      = rng.normal(0, 0.04, n)
    noise_2      = rng.normal(0, 0.3, n)
    dp           = np.clip(health * power * (1 + noise_1) + 0.6 + noise_2, 0, None)

    tables.append(pd.DataFrame({"unit_id": unit_id,
                                "ts": start + pd.to_timedelta(seconds, unit="s"),
                                "running": running > 0.5,
                                "current_a": np.round(current, 3),
                                "voltage_v": np.round(voltage, 2),
                                "dp": np.round(dp, 3),
                                "temp_c": np.round(temp, 2)}))
raw_table = pd.concat(tables, ignore_index=True)
print()
print(pd.DataFrame({"rows": [len(raw_table)]}, index=["raw data rows"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the tables

# COMMAND ----------

rows = []
for unit in units:
    rows.append({"unit_id": unit["unit_id"], "site": unit["site"], "customer": unit["customer"],
                 "commissioned": date_at(unit["unit_id"], 0.0).round("D")})
units_table = pd.DataFrame(rows)

def write_table(table, name):
    writer = spark.createDataFrame(table).write.mode("overwrite")
    writer.option("overwriteSchema", "true").saveAsTable(CATALOG + "." + SCHEMA + "." + name)
    print(name, table.shape)

write_table(units_table, "units")
write_table(period_table, "telemetry_period")
write_table(raw_table, "telemetry_raw")
write_table(reports, "service_reports")
write_table(actual_replacements, "actual_replacements")
write_table(actual_life, "actual_remaining_life")
