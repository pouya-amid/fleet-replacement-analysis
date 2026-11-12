# Databricks notebook source
# MAGIC %md
# MAGIC # Separating failures from routine replacements in fleet data
# MAGIC
# MAGIC Pouya Amid, November 2026
# MAGIC
# MAGIC - This notebook shows how I separate failure replacements from routine ones when the service reports don't say which is which. It then forecasts failures per quarter.
# MAGIC - The original work was done for my employer and is confidential. I rebuilt the approach here on simulated data for 250 units.
# MAGIC - Sections 2 and 3 cover the health factor and the early warning. Sections 4 and 5 find the replacements and separate failures from non-failures. Section 6 checks the counts. Section 7 gives the forecast.
# MAGIC - The checks behind the thresholds and the forecast range are in a second notebook, `02_detailed_checks`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Terms
# MAGIC - **Health factor**: differential pressure divided by power, calculated on every data row.
# MAGIC - **Period**: 20 operating hours. The health factor of a period is the average of its row values.
# MAGIC - **Loss**: how far the health factor has fallen below the value of a new part. A new part reads within a few percent of that value, above or below it. So the loss of a new part is close to 0 and can be slightly negative.
# MAGIC - **Replacement level**: a loss of 0.50 (50%). A part is normally replaced there.
# MAGIC - **Failure (A)**: a replaced part that had lost 40% or more. It had failed or was about to fail.
# MAGIC - **Non-failure (B)**: a replaced part that had lost less than 40%.
# MAGIC - **Actual**: the data is simulated, so the actual events are known. They are used only to check results.

# COMMAND ----------

# MAGIC %pip install -q -U xgboost threadpoolctl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data
# MAGIC - 250 units, with records from June 2022 to September 2026. A unit runs about 40 operating hours a week and has between 1,000 and 8,000 operating hours.
# MAGIC - The data is made by `00_simulate`. Run that notebook first.
# MAGIC - Four tables: the units, the health factor of every period, a sample of raw data rows, and the service reports.
# MAGIC - A service report has a date, the reason written down, and the inspected class if the part was inspected.
# MAGIC - The actual replacements are read here too. They are used only to check results.

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCHEMA            = "workspace.pge_demo"
PERIOD_HOURS      = 20       # operating hours in one period
REPLACEMENT_LEVEL = 0.50     # loss at which a part is normally replaced
MINIMUM_LOAD_KW   = 20.0     # below this the unit is starting, stopping or idle
CLASS_WORDS       = {"A": "failure (A)", "B": "non-failure (B)"}

plt.rcParams.update({"figure.figsize": (10, 5.6), "font.size": 14, "savefig.dpi": 200})
os.makedirs("charts", exist_ok=True)
random_numbers = np.random.default_rng(0)

def read_table(name, order):
    table = spark.table(SCHEMA + "." + name).toPandas()
    table = table.sort_values(order).reset_index(drop=True)
    return table

def save_table(table, name):
    table = table.copy()
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].fillna("")
    writer = spark.createDataFrame(table).write.mode("overwrite")
    writer.option("overwriteSchema", "true").saveAsTable(SCHEMA + "." + name)

units   = read_table("units", ["unit_id"])
periods = read_table("telemetry_period", ["unit_id", "op_hours_cum"])
raw     = read_table("telemetry_raw", ["unit_id", "ts"])
reports = read_table("service_reports", ["unit_id", "op_hours_at_service"])
actual  = read_table("actual_replacements", ["unit_id", "op_hours"])     # only to check results

sizes = pd.DataFrame({"rows": [len(units), len(periods), len(raw), len(reports)]},
                     index=["units", "periods of 20 operating hours",
                            "raw data rows, 5 units, 6 hours each", "service reports"])
print(sizes.to_string())

reasons = reports.reason.fillna("reason left blank").value_counts()
print()
print(reasons.to_frame("service reports").to_string())

inspections = reports.inspected_class.map(CLASS_WORDS).fillna("not inspected").value_counts()
print()
print(inspections.to_frame("service reports").to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Health factor
# MAGIC - Power is not recorded. On every raw data row it is current times voltage.
# MAGIC - Health factor of a row: differential pressure divided by power. Only rows where the unit is running above the minimum load of 20 kW are used.
# MAGIC - Health factor of a period: the average of its row values. The period table holds this average for every unit.
# MAGIC - Baseline: the fleet median over the first 10 periods of every unit. There is one unit type, so one baseline.
# MAGIC - Loss: 1 minus health factor divided by baseline. A new part is at 0 and the replacement level is 0.50.

# COMMAND ----------

raw["power_kw"]      = raw.current_a * raw.voltage_v / 1000.0
raw["above_minimum"] = raw.running & (raw.power_kw >= MINIMUM_LOAD_KW)
raw["health_factor"] = raw.dp / raw.power_kw
raw.loc[~raw.above_minimum, "health_factor"] = np.nan      # not used below the minimum load

# one row every half hour, from the first unit in the sample
first_unit = raw[raw.unit_id == raw.unit_id.iloc[0]]
some_rows  = first_unit.iloc[::3600]
some_rows  = some_rows[["ts", "current_a", "voltage_v", "power_kw", "dp", "above_minimum",
                        "health_factor"]]
some_rows.columns = ["time", "current (A)", "voltage (V)", "power (kW)",
                     "differential pressure", "above minimum load", "health factor"]
print(some_rows.round(3).to_string(index=False))

rows = []
for unit_id in raw.unit_id.unique():
    unit_rows = raw[raw.unit_id == unit_id]
    rows.append({"unit": unit_id,
                 "data rows": len(unit_rows),
                 "rows above minimum load": int(unit_rows.above_minimum.sum()),
                 "health factor, average of rows": unit_rows.health_factor.mean()})
print()
print(pd.DataFrame(rows).round(3).to_string(index=False))

# COMMAND ----------

first_ten_periods = periods[periods.period_id < 10]
baseline          = first_ten_periods.health_factor.median()
periods["loss"]   = 1.0 - periods.health_factor / baseline

# the periods of each unit, in time order
periods_by_unit = {}
for unit_id in units.unit_id:
    periods_by_unit[unit_id] = periods[periods.unit_id == unit_id]

# operating hours at the end of each unit's record
record_end = periods.groupby("unit_id").op_hours_cum.max()

# example for the chart: the inspected failure with the longest record after it
failures = reports[reports.inspected_class == "A"].copy()
failures["hours_after"] = failures.unit_id.map(record_end) - failures.op_hours_at_service
example  = failures.sort_values("hours_after").iloc[-1]

fig, ax = plt.subplots()
for unit_id in units.unit_id:
    table = periods_by_unit[unit_id]
    ax.plot(table.op_hours_cum, 1.0 - table.loss, color="0.8", linewidth=0.5)
table = periods_by_unit[example.unit_id]
ax.plot(table.op_hours_cum, 1.0 - table.loss, color="C0", linewidth=2,
        label="unit " + example.unit_id)
ax.axvline(example.op_hours_at_service, color="k", linestyle="--", linewidth=1.5,
           label="reported replacement")
ax.axhline(1.0 - REPLACEMENT_LEVEL, color="C3", linestyle=":", linewidth=2.5,
           label="replacement level")
ax.set_ylim(0.35, 1.15)
ax.set_xlabel("operating hours on the unit")
ax.set_ylabel("health factor (1.0 = new)")
ax.set_title("Health factor of one unit, the fleet in grey")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("charts/health_factor.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Early warning
# MAGIC - Average loss: the average of the loss over the last 10 periods, about 200 operating hours. It looks backward only: the period itself and the 9 before it.
# MAGIC - Why an average and not single periods: one period scatters by about 0.03 of loss, from load changes and sensor noise. A part that is really at 0.25 can show one reading of 0.30. The average follows the wear and not the scatter.
# MAGIC - A warning is raised the first time the average reaches 0.30.
# MAGIC - Why 0.30 and 10 periods: both were tested in `02_detailed_checks`. A lower level or a shorter average warns earlier, but more of the warnings come on parts that are still far from the replacement level. A higher level or a longer average misses failures.
# MAGIC - About 3% of periods have no data, so 10 periods can cover a little more than 200 operating hours.
# MAGIC - The chart shows one unit: the loss of each period, the average, and the hour at which the warning is raised.

# COMMAND ----------

SMOOTH_PERIODS          = 10      # periods in the average. 10 periods is 200 operating hours
EARLY_WARNING_THRESHOLD = 0.30    # average loss that raises the warning
LEAD_HOURS              = 400     # a warning counts only if it comes this long before the replacement
SETTLE_HOURS            = 2000    # a warning on a part that then runs this long is a false alarm

def average_of_last_periods(loss):
    # for every period: the average loss of that period and the 9 periods before it.
    # it looks backward only. the first 9 periods of a part have no value.
    average = np.full(len(loss), np.nan)
    for i in range(SMOOTH_PERIODS - 1, len(loss)):
        average[i] = loss[i - SMOOTH_PERIODS + 1 : i + 1].mean()
    return average

def first_warning(part_periods, threshold):
    # operating hours at which the average first reaches the threshold. NaN if it never does.
    average = average_of_last_periods(part_periods.loss.values)
    hours   = part_periods.op_hours_cum.values
    for i in range(len(average)):
        if average[i] >= threshold:
            return float(hours[i])
    return np.nan

# the same unit as in section 2: the loss of every period, and the average
table   = periods_by_unit[example.unit_id]
part    = table[table.op_hours_cum <= example.op_hours_at_service]
average = average_of_last_periods(part.loss.values)
warned  = first_warning(part, EARLY_WARNING_THRESHOLD)

fig, ax = plt.subplots()
ax.plot(part.op_hours_cum, part.loss, color="0.6", linewidth=1, label="loss of each period")
ax.plot(part.op_hours_cum, average, color="C0", linewidth=2.5,
        label="average of the last " + str(SMOOTH_PERIODS) + " periods")
ax.axhline(EARLY_WARNING_THRESHOLD, color="C1", linestyle="--", linewidth=2, label="warning level")
ax.axhline(REPLACEMENT_LEVEL, color="C3", linestyle=":", linewidth=2.5, label="replacement level")
if not np.isnan(warned):
    ax.axvline(warned, color="C1", linewidth=1.5, label="warning raised")
ax.axvline(example.op_hours_at_service, color="k", linestyle="--", linewidth=1.5,
           label="reported replacement")
ax.set_xlabel("operating hours on the unit")
ax.set_ylabel("loss")
ax.set_title("Early warning on unit " + example.unit_id)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("charts/early_warning.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **The test cases**
# MAGIC - One case for every inspected replacement: the part that came out. The inspection says whether it was a failure.
# MAGIC - Units with no service report: one case for every actual replacement, or one case for the whole record if the part was never replaced.
# MAGIC - Part start: the operating hours at which the part went in. 0 for the first part of a unit.

# COMMAND ----------

cases = []

# one case for every inspected replacement
inspected_reports = reports[reports.inspected_class.notna()]
for i, report in inspected_reports.iterrows():
    earlier = reports[(reports.unit_id == report.unit_id)
                      & (reports.op_hours_at_service < report.op_hours_at_service)]
    if len(earlier) > 0:
        part_start = earlier.op_hours_at_service.max()
    else:
        part_start = 0.0
    if report.inspected_class == "A":
        kind = "failure"
    else:
        kind = "non-failure"
    cases.append({"unit_id": report.unit_id, "kind": kind, "source": "inspected",
                  "part_start": part_start, "replaced_at": report.op_hours_at_service})

# units with no service report
units_with_report = list(reports.unit_id)
for unit_id in units.unit_id:
    if unit_id in units_with_report:
        continue
    unit_actual = actual[actual.unit_id == unit_id]
    part_start  = 0.0
    for i, replaced in unit_actual.iterrows():
        if replaced.actual_class == "A":
            kind = "failure"
        else:
            kind = "non-failure"
        cases.append({"unit_id": unit_id, "kind": kind, "source": "no report",
                      "part_start": part_start, "replaced_at": replaced.op_hours})
        part_start = replaced.op_hours
    if len(unit_actual) == 0:
        cases.append({"unit_id": unit_id, "kind": "never replaced", "source": "",
                      "part_start": 0.0, "replaced_at": np.nan})

cases = pd.DataFrame(cases)
print(cases.groupby(["kind", "source"]).size().to_frame("cases").to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC **The test**
# MAGIC - Replaced part: only its periods up to 400 operating hours before the replacement are used. A warning later than that comes too late to count.
# MAGIC   - Failure with a warning: caught. Lead time is the hours from the warning to the replacement.
# MAGIC   - Failure with no warning: missed.
# MAGIC   - Non-failure with a warning: the warning came before a scheduled change. Neither a catch nor a false alarm.
# MAGIC - Never replaced, with a warning: a false alarm if the unit then ran 2,000 more operating hours. With less record after the warning it is too recent to judge.
# MAGIC - Result: all 24 failures in the test are caught, typically 1,015 operating hours ahead, which is about 25 weeks of operation. No warning has proved false.
# MAGIC - 75 warnings are on parts that are not replaced yet, so they are too recent to judge. `02_detailed_checks` compares them with the simulation: 66 are within 2,000 operating hours of the replacement level and 9 are further away.

# COMMAND ----------

rows = []
for i, case in cases.iterrows():
    unit_periods = periods_by_unit[case.unit_id]

    if case.kind == "never replaced":
        part_periods = unit_periods
    else:
        last_hours   = case.replaced_at - LEAD_HOURS
        part_periods = unit_periods[(unit_periods.op_hours_cum > case.part_start)
                                    & (unit_periods.op_hours_cum <= last_hours)]

    warning_hours = first_warning(part_periods, EARLY_WARNING_THRESHOLD)
    has_warning   = not np.isnan(warning_hours)
    lead_hours    = np.nan

    if case.kind == "failure":
        if has_warning:
            outcome    = "caught"
            lead_hours = case.replaced_at - warning_hours
        else:
            outcome = "missed"
    elif case.kind == "non-failure":
        if has_warning:
            outcome = "warning before a scheduled change"
        else:
            outcome = "no warning"
    else:
        if not has_warning:
            outcome = "no warning"
        elif unit_periods.op_hours_cum.max() - warning_hours >= SETTLE_HOURS:
            outcome = "false alarm"
        else:
            outcome = "too recent to judge"

    rows.append({"outcome": outcome, "warning_hours": warning_hours, "lead_hours": lead_hours})

results = pd.DataFrame(rows)
cases["outcome"]       = results.outcome
cases["warning_hours"] = results.warning_hours
cases["lead_hours"]    = results.lead_hours
save_table(cases, "early_warning_cases")

by_kind = pd.crosstab(cases.kind, cases.outcome)
by_kind.index.name   = None
by_kind.columns.name = None
print(by_kind.to_string())

caught       = int((cases.outcome == "caught").sum())
missed       = int((cases.outcome == "missed").sum())
false_alarms = int((cases.outcome == "false alarm").sum())
too_recent   = int((cases.outcome == "too recent to judge").sum())
typical_lead = cases[cases.outcome == "caught"].lead_hours.median()

summary = pd.DataFrame({"count": [caught, missed, false_alarms, too_recent, round(typical_lead)]},
                       index=["failures caught", "failures missed", "false alarms",
                              "never replaced, warning too recent to judge",
                              "typical lead time, operating hours (median)"])
print()
print(summary.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Finding replacements
# MAGIC - At every period, the 10 periods before it are compared with the 3 periods after it. The period itself is skipped, because a replacement can fall inside it.
# MAGIC - Step: average loss before minus average loss after.
# MAGIC - Scatter: the standard deviation of the loss over the 10 periods before.
# MAGIC - Four factors. Each runs from 0 to 1 between two values:
# MAGIC   - Size of the step: 0 at a step of 0.10, 1 at 0.20. A part is not changed before it has lost 20%.
# MAGIC   - Step against scatter: the step divided by the scatter. 0 at 3 times the scatter, 1 at 6 times.
# MAGIC   - Clear of the scatter: the best of the 10 periods before minus the worst of the 3 periods after. 0 where they overlap, 1 where they are 0.10 apart.
# MAGIC   - Out of service: idle days around the period, the calendar time that operating hours do not explain. 0 with none, 1 at 4 days.
# MAGIC - Index: the sum of the four factors, from 0 to 4.
# MAGIC - The rate of loss before is also kept: the slope of a straight line through up to 25 periods before, per 1,000 operating hours. It is not in the index. Section 5 uses it.
# MAGIC - A replacement can be found once 3 full periods follow it: 60 to 80 operating hours.
# MAGIC - The table shows how many periods reach an index of 1, 2 and 3.

# COMMAND ----------

BEFORE_PERIODS = 10      # periods before the one being tested
AFTER_PERIODS  = 3       # periods after it
RATE_PERIODS   = 25      # periods used for the rate of loss before

def to_factor(value, zero_at, one_at):
    # 0 at or below zero_at, 1 at or above one_at, a straight line in between
    factor = (value - zero_at) / (one_at - zero_at)
    return min(max(factor, 0.0), 1.0)

def factors_of_unit(unit_periods):
    table = unit_periods.reset_index(drop=True).copy()
    n     = len(table)
    loss  = table.loss.values
    hours = table.op_hours_cum.values.astype(float)

    # calendar time of every period, in days from the start of the unit's record
    first_day = table.ts_start.iloc[0]
    start_day = ((table.ts_start - first_day).dt.total_seconds() / 86400).values
    end_day   = ((table.ts_end - first_day).dt.total_seconds() / 86400).values

    names = ["loss_before", "loss_after", "step", "scatter", "step_vs_scatter", "clear_gap",
             "days_out", "loss_rate_before", "factor_step", "factor_scatter", "factor_clear",
             "factor_out_of_service"]
    values = {}
    for name in names:
        values[name] = np.full(n, np.nan)
    change_index = np.zeros(n)

    # the first 10 and the last 3 periods of the record cannot be tested
    for i in range(BEFORE_PERIODS, n - AFTER_PERIODS):
        before = loss[i - BEFORE_PERIODS : i]              # the 10 periods before period i
        after  = loss[i + 1 : i + 1 + AFTER_PERIODS]       # the 3 periods after it

        loss_before = before.mean()
        loss_after  = after.mean()
        step        = loss_before - loss_after
        scatter     = before.std(ddof=1)
        clear_gap   = before.min() - after.max()

        # idle days: calendar days from the end of the period before to the start of the period
        # after, less the days that the operating hours in between normally take
        calendar_days   = start_day[i + 1] - end_day[i - 1]
        periods_between = (hours[i + 1] - hours[i - 1]) / PERIOD_HOURS - 1
        days_per_period = end_day[i] - start_day[i]
        days_out        = calendar_days - periods_between * days_per_period

        # rate of loss before: a straight line through up to 25 periods before
        first = max(0, i - RATE_PERIODS)
        line  = np.polyfit(hours[first:i] - hours[i], loss[first:i], 1)
        loss_rate_before = 1000 * line[0]

        factor_step           = to_factor(step, 0.10, 0.20)
        factor_scatter        = to_factor(step / scatter, 3.0, 6.0)
        factor_clear          = to_factor(clear_gap, 0.0, 0.10)
        factor_out_of_service = to_factor(days_out, 0.0, 4.0)

        values["loss_before"][i]           = loss_before
        values["loss_after"][i]            = loss_after
        values["step"][i]                  = step
        values["scatter"][i]               = scatter
        values["step_vs_scatter"][i]       = step / scatter
        values["clear_gap"][i]             = clear_gap
        values["days_out"][i]              = days_out
        values["loss_rate_before"][i]      = loss_rate_before
        values["factor_step"][i]           = factor_step
        values["factor_scatter"][i]        = factor_scatter
        values["factor_clear"][i]          = factor_clear
        values["factor_out_of_service"][i] = factor_out_of_service
        change_index[i] = factor_step + factor_scatter + factor_clear + factor_out_of_service

    for name in names:
        table[name] = values[name]
    table["change_index"] = change_index
    return table

tables = []
for unit_id in units.unit_id:
    tables.append(factors_of_unit(periods_by_unit[unit_id]))
factors = pd.concat(tables, ignore_index=True)
save_table(factors, "period_factors")

at_or_above = pd.DataFrame({"periods": [len(factors),
                                        int((factors.change_index >= 1).sum()),
                                        int((factors.change_index >= 2).sum()),
                                        int((factors.change_index >= 3).sum())]},
                           index=["all periods", "index 1 or more", "index 2 or more",
                                  "index 3 or more"])
print(at_or_above.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC **From index to replacements**
# MAGIC - Candidate: a period where the index reaches the threshold. Where neighbouring periods also reach it, only the one with the highest index is kept. One replacement gives one candidate, dated at the step.
# MAGIC - Threshold table: for each threshold, the service reports that have a candidate within 500 operating hours, and the candidates with no report nearby. About one replacement in ten has no report.
# MAGIC - Threshold: 2.0, half of the scale. It is the highest threshold that still finds every service report. The result does not depend on the exact value: from 2.0 to 3.0 the number of candidates goes from 37 to 35.

# COMMAND ----------

KEEP_APART_HOURS = 200    # two candidates on one unit are at least this far apart
MATCH_HOURS      = 500    # closer than this on the same unit: the same replacement

def find_candidates(threshold):
    kept_rows = []
    above = factors[factors.change_index >= threshold]
    for unit_id in above.unit_id.unique():
        # highest index first. a period close to one already kept is the same replacement
        unit_rows  = above[above.unit_id == unit_id]
        unit_rows  = unit_rows.sort_values(["change_index", "step"], ascending=False)
        kept_hours = []
        for row_number, row in unit_rows.iterrows():
            too_close = False
            for hours in kept_hours:
                if abs(row.op_hours_cum - hours) <= KEEP_APART_HOURS:
                    too_close = True
            if not too_close:
                kept_rows.append(row_number)
                kept_hours.append(row.op_hours_cum)
    candidates = factors.loc[kept_rows].sort_values(["unit_id", "op_hours_cum"])
    return candidates.reset_index(drop=True)

def nearest_on_unit(unit_id, hours, table, hours_column):
    # row number of the row of table that is on the same unit and nearest in operating hours.
    # None if there is none within 500 hours.
    best_row   = None
    best_apart = MATCH_HOURS
    for row_number, row in table[table.unit_id == unit_id].iterrows():
        apart = abs(row[hours_column] - hours)
        if apart <= best_apart:
            best_row   = row_number
            best_apart = apart
    return best_row

rows = []
for threshold in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    candidates    = find_candidates(threshold)
    reports_found = 0
    with_report   = []           # candidates that have a service report nearby
    for i, report in reports.iterrows():
        row_number = nearest_on_unit(report.unit_id, report.op_hours_at_service,
                                     candidates, "op_hours_cum")
        if row_number is not None:
            reports_found = reports_found + 1
            if row_number not in with_report:
                with_report.append(row_number)
    rows.append({"threshold": threshold,
                 "candidates": len(candidates),
                 "service reports found": reports_found,
                 "service reports not found": len(reports) - reports_found,
                 "candidates with no report nearby": len(candidates) - len(with_report)})
print(pd.DataFrame(rows).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC **The replacements found**
# MAGIC - Hours on the part: counted from the replacement found before it on the same unit, or from the start of the unit.
# MAGIC - Each service report is given to the nearest replacement found on the same unit, within 500 operating hours.
# MAGIC - Measurements of a replacement found: loss before, step, step against scatter, rate of loss before, and hours on the part.

# COMMAND ----------

INDEX_THRESHOLD = 2.0
MEASUREMENTS    = ["loss_before", "step", "step_vs_scatter", "loss_rate_before", "hours_on_part"]

replacements = find_candidates(INDEX_THRESHOLD)

# when the part that came out had gone in
part_starts = []
for i, found in replacements.iterrows():
    earlier = replacements[(replacements.unit_id == found.unit_id)
                           & (replacements.op_hours_cum < found.op_hours_cum)]
    if len(earlier) > 0:
        part_starts.append(earlier.op_hours_cum.max())
    else:
        part_starts.append(0.0)
replacements["part_start"]    = part_starts
replacements["hours_on_part"] = replacements.op_hours_cum - replacements.part_start

# the service report of each replacement found, where there is one
replacements["reason"]          = None
replacements["inspected_class"] = None
for i, report in reports.iterrows():
    row_number = nearest_on_unit(report.unit_id, report.op_hours_at_service,
                                 replacements, "op_hours_cum")
    if row_number is not None:
        replacements.loc[row_number, "reason"]          = report.reason
        replacements.loc[row_number, "inspected_class"] = report.inspected_class

shown = replacements[["unit_id", "op_hours_cum"] + MEASUREMENTS + ["days_out", "change_index"]].copy()
shown["inspected class"] = replacements.inspected_class.map(CLASS_WORDS).fillna("not inspected")
shown.columns = ["unit", "operating hours", "loss before", "step", "step against scatter",
                 "rate of loss before", "hours on the part", "days out of service", "index",
                 "inspected class"]
print(shown.head(10).round(2).to_string(index=False))
print()
print(pd.DataFrame({"count": [len(replacements),
                              int(replacements.inspected_class.notna().sum())]},
                   index=["replacements found", "of these, inspected"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Failure or non-failure
# MAGIC - Each replacement found is described by its five measurements. Days out of service is left out here: it describes the visit, not the part.
# MAGIC - Inspected: after some replacements the part that came out was inspected and classed as failure (A) or non-failure (B). These are the only confirmed classes.
# MAGIC - K-means with 2 groups on the standardized measurements. It is not given the inspection results.
# MAGIC - Cross-tab: the two groups against the inspected class.
# MAGIC - XGBoost is trained on the inspected replacements.
# MAGIC - Why both: the grouping shows that the measurements alone separate two kinds of replacement. The classifier is the step that labels each new replacement automatically.
# MAGIC - Why a test is needed: a model scores too well on the parts it was trained on. An honest score needs parts the model has not seen.
# MAGIC - Test, 5 folds: 29 inspected parts are too few to set a test group aside once. So they are split in 5 groups, and each group is labelled by a model trained on the other 4. Every part is tested once, by a model that did not see it.
# MAGIC - There is no separate validation group, because no setting of the model is tuned on these results.
# MAGIC - The model trained on all inspected replacements then labels every replacement found, inspected or not.
# MAGIC - Is the model overfit? The checks are in `02_detailed_checks`, section 5. On the parts it was trained on it is right on 29 of 29, and on parts it did not see on 28 of 29. The 8 replacements that were not inspected are all labelled correctly.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler       = StandardScaler()
standardized = scaler.fit_transform(replacements[MEASUREMENTS])
kmeans       = KMeans(n_clusters=2, n_init=10, random_state=0)
group_number = kmeans.fit_predict(standardized)

# group 1 is the one with the lower average loss before
loss_of_group_0 = replacements.loss_before[group_number == 0].mean()
loss_of_group_1 = replacements.loss_before[group_number == 1].mean()
if loss_of_group_0 < loss_of_group_1:
    low_group = 0
else:
    low_group = 1
replacements["group"] = "group 2"
replacements.loc[group_number == low_group, "group"] = "group 1"

inspected_words = replacements.inspected_class.map(CLASS_WORDS).fillna("not inspected")
cross_tab = pd.crosstab(replacements.group, inspected_words, margins=True, margins_name="total")
cross_tab.index.name   = None
cross_tab.columns.name = None
print(cross_tab.to_string())

fig, ax = plt.subplots()
chosen = replacements[inspected_words == "not inspected"]
ax.scatter(chosen.loss_before, chosen.step, s=70, color="0.6", label="not inspected")
chosen = replacements[inspected_words == "non-failure (B)"]
ax.scatter(chosen.loss_before, chosen.step, s=70, color="C0", label="non-failure (B)")
chosen = replacements[inspected_words == "failure (A)"]
ax.scatter(chosen.loss_before, chosen.step, s=70, color="C3", label="failure (A)")
ax.set_xlabel("loss before the replacement")
ax.set_ylabel("size of the step")
ax.set_title("Replacements found, by inspected class")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("charts/two_groups.png")
plt.show()

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

def new_model():
    return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                         eval_metric="logloss", random_state=0)

inspected  = replacements[replacements.inspected_class.notna()]
inputs     = inspected[MEASUREMENTS].values
is_failure = (inspected.inspected_class == "A").astype(int).values

# 5-fold test: every inspected replacement is labelled by a model that did not see it
folds      = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
test_label = np.zeros(len(inspected), dtype=int)
for train_rows, test_rows in folds.split(inputs, is_failure):
    model = new_model()
    model.fit(inputs[train_rows], is_failure[train_rows])
    test_label[test_rows] = model.predict(inputs[test_rows])

inspected_as = inspected.inspected_class.map(CLASS_WORDS).values
labelled_as  = np.where(test_label == 1, "labelled failure (A)", "labelled non-failure (B)")
test_table   = pd.crosstab(inspected_as, labelled_as)
test_table.index        = "inspected " + test_table.index
test_table.index.name   = None
test_table.columns.name = None
print(test_table.to_string())

agree = int((test_label == is_failure).sum())
print()
print(pd.DataFrame({"count": [len(inspected), agree, len(inspected) - agree]},
                   index=["inspected replacements in the test",
                          "label agrees with the inspection",
                          "label disagrees"]).to_string())

# the model trained on all inspected replacements labels every replacement found
model = new_model()
model.fit(inputs, is_failure)
predicted = model.predict(replacements[MEASUREMENTS].values)
replacements["label"] = np.where(predicted == 1, "A", "B")

replacements["test_label"] = None
replacements.loc[inspected.index, "test_label"] = np.where(test_label == 1, "A", "B")

labels = replacements.label.map(CLASS_WORDS).value_counts().to_frame("replacements found")
print()
print(labels.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. How the counts add up
# MAGIC - Every actual replacement is paired with its service report and with the replacement found by the method: the nearest one on the same unit, within 500 operating hours.
# MAGIC - Table 1: found or not found by the method, against inspected or not inspected. Not inspected means a report without an inspection, or no report.
# MAGIC - Table 2: the same replacements counted three ways.
# MAGIC   - Service reports: by the reason written down. A performance reason counts as failure (A), a scheduled reason as non-failure (B).
# MAGIC   - Method: by the label from section 5.
# MAGIC   - Actual: by the actual class.
# MAGIC - Then two counts and a list: replacements found that pair with no actual replacement, actual replacements not found, and each one not found with the reason.

# COMMAND ----------

PERFORMANCE_REASONS = ["low output", "performance below spec", "unit derated", "customer complaint"]

replacements["actual_class"] = None
rows = []
for i, replaced in actual.iterrows():
    report_row = nearest_on_unit(replaced.unit_id, replaced.op_hours, reports,
                                 "op_hours_at_service")
    found_row  = nearest_on_unit(replaced.unit_id, replaced.op_hours, replacements,
                                 "op_hours_cum")

    # what the service report says
    was_inspected = False
    report_hours  = np.nan
    if report_row is None:
        reports_say = "no report"
    else:
        report        = reports.loc[report_row]
        report_hours  = report.op_hours_at_service
        was_inspected = report.inspected_class in ["A", "B"]
        if report.reason is None or pd.isna(report.reason):
            reports_say = "reason left blank"
        elif report.reason in PERFORMANCE_REASONS:
            reports_say = "failure (A)"
        else:
            reports_say = "non-failure (B)"

    # what the method says
    found          = found_row is not None
    found_at_hours = np.nan
    if found:
        found_at_hours = replacements.op_hours_cum[found_row]
        method_says    = CLASS_WORDS[replacements.label[found_row]]
        replacements.loc[found_row, "actual_class"] = replaced.actual_class
    else:
        method_says = "not found"

    rows.append({"found": found, "found_row": found_row, "inspected": was_inspected,
                 "report_hours": report_hours, "found_at_hours": found_at_hours,
                 "reports_say": reports_say, "method_says": method_says,
                 "actually": CLASS_WORDS[replaced.actual_class]})

accounting = pd.concat([actual, pd.DataFrame(rows)], axis=1)
accounting["hours_after"] = accounting.unit_id.map(record_end) - accounting.op_hours

# table 1
found_words     = np.where(accounting.found, "found by the method", "not found")
inspected_words = np.where(accounting.inspected, "inspected", "not inspected")
table_1 = pd.crosstab(found_words, inspected_words, margins=True, margins_name="total")
table_1.index.name   = None
table_1.columns.name = None
print(table_1.to_string())

# table 2
table_2 = pd.DataFrame({"service reports": accounting.reports_say.value_counts(),
                        "method":          accounting.method_says.value_counts(),
                        "actual":          accounting.actually.value_counts()})
table_2 = table_2.reindex(["failure (A)", "non-failure (B)", "reason left blank", "no report",
                           "not found"])
table_2 = table_2.fillna(0).astype(int)
table_2.loc["total"] = table_2.sum()
print()
print(table_2.to_string())

# replacements found that pair with no actual replacement
paired_rows = list(accounting.found_row.dropna())
unpaired    = 0
for row_number in replacements.index:
    if row_number not in paired_rows:
        unpaired = unpaired + 1
print()
print(pd.DataFrame({"count": [unpaired, int((~accounting.found).sum())]},
                   index=["found by the method, no actual replacement",
                          "actual replacements not found"]).to_string())

# every actual replacement not found, with the reason
rows = []
for i, missed_one in accounting[~accounting.found].iterrows():
    unit_factors = factors[factors.unit_id == missed_one.unit_id]
    nearby = unit_factors[(unit_factors.op_hours_cum - missed_one.op_hours).abs()
                          <= 3 * PERIOD_HOURS]
    if missed_one.hours_after < (AFTER_PERIODS + 1) * PERIOD_HOURS:
        why = "record ends too soon after it"
    else:
        why = "index below the threshold"
    rows.append({"unit": missed_one.unit_id,
                 "operating hours": missed_one.op_hours,
                 "actual class": missed_one.actually,
                 "loss at replacement": missed_one.loss_at,
                 "days out of service": missed_one.days_out,
                 "hours of record after": missed_one.hours_after,
                 "highest index nearby": nearby.change_index.max(),
                 "reason not found": why})
if len(rows) > 0:
    print()
    print(pd.DataFrame(rows).round(2).to_string(index=False))

# COMMAND ----------

sources = ["service reports", "method", "actual"]
others  = table_2.loc[["reason left blank", "no report", "not found"], sources].sum()
x       = np.arange(len(sources))

fig, ax = plt.subplots()
bars = ax.bar(x - 0.27, table_2.loc["failure (A)", sources], width=0.26, color="C3",
              label="failure (A)")
ax.bar_label(bars)
bars = ax.bar(x, table_2.loc["non-failure (B)", sources], width=0.26, color="C0",
              label="non-failure (B)")
ax.bar_label(bars)
bars = ax.bar(x + 0.27, others, width=0.26, color="0.7",
              label="reason left blank, no report, or not found")
ax.bar_label(bars)
ax.set_xticks(x)
ax.set_xticklabels(sources)
ax.set_ylabel("replacements")
ax.set_ylim(0, table_2.loc["total", "actual"])
ax.set_title("The same replacements, counted three ways")
ax.legend(loc="upper center")
fig.tight_layout()
fig.savefig("charts/counts.png")
plt.show()

save_table(replacements, "replacements_found")
save_table(accounting.drop(columns=["found_row"]), "replacement_accounting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Failures due per quarter
# MAGIC - The forecast starts from the part now fitted in every unit. The question for each part: how many operating hours are left before it reaches the replacement level?
# MAGIC - Rule (a), the trend: for a part with more than 2,000 operating hours that is losing steadily. A straight line is fitted through its last 50 periods, about 1,000 operating hours, and extended until it reaches the replacement level. Losing steadily means the line rises by at least 0.06 per 1,000 operating hours.
# MAGIC - Rule (b), the life table: for the other parts, where the trend is still too flat to extend. Hours left are the expected life for the type of part, minus the hours already on the part.
# MAGIC - The life table already exists and is taken as given. It has a life for a new install, for a part fitted at a failure (A) replacement and for a part fitted at a non-failure (B) replacement.
# MAGIC - Why the type matters: a part fitted on its own after a failure goes into a unit whose other parts are worn, and it lasts shorter. A part fitted along with other parts at scheduled work lasts longer. This is where the labels from section 5 are used.
# MAGIC - Test of rule (a): on the failure replacements found, the last 1,000 hours of the part are hidden and the rule is applied to what is left. Share: the 1,000 hours the part really had left, divided by the hours the rule predicted. A share below 1 means the part reached the level sooner than predicted.
# MAGIC - The 10% and 90% shares from this test give the range of rule (a).

# COMMAND ----------

HOURS_FOR_TREND   = 2000     # rule (a) needs more than this many hours on the part
PERIODS_FOR_TREND = 50       # periods in the straight line, about the last 1,000 operating hours
MINIMUM_RATE      = 0.06     # loss per 1,000 hours. below this the trend is too flat to extend
HIDDEN_HOURS      = 1000     # hours hidden in the test of rule (a)
HORIZON_HOURS     = 6000     # hours remaining are held between 0 and this
RANGE_IN_SD       = 2.563    # width of the 10% to 90% range of a normal distribution, in standard deviations

PART_TYPE = {"A": "fitted at a failure (A) replacement",
             "B": "fitted at a non-failure (B) replacement"}

life_table = pd.DataFrame({"expected": [5800, 5000, 6000],
                           "low":      [4300, 3500, 4500],
                           "high":     [7300, 6500, 7500]},
                          index=["new install", PART_TYPE["A"], PART_TYPE["B"]])
print(life_table.to_string())

def hours_to_level(part_periods):
    # rule (a): a straight line through the last 50 periods of the part,
    # extended until it reaches the replacement level
    recent   = part_periods.tail(PERIODS_FOR_TREND)
    hours    = recent.op_hours_cum.values - recent.op_hours_cum.values[-1]    # 0 at the last period
    line     = np.polyfit(hours, recent.loss.values, 1)
    rate     = line[0]                                 # loss per operating hour
    loss_now = line[1]                                 # the line at the last period
    if rate * 1000 < MINIMUM_RATE:
        return np.nan
    return (REPLACEMENT_LEVEL - loss_now) / rate

# test of rule (a) on the failure replacements found: hide the last 1,000 hours of the part,
# apply the rule to what is left, and compare with the 1,000 hours the part really had
rows = []
for i, found in replacements[replacements.label == "A"].iterrows():
    unit_periods = periods_by_unit[found.unit_id]
    part_end     = found.op_hours_cum - PERIOD_HOURS
    last_visible = part_end - HIDDEN_HOURS
    visible      = unit_periods[(unit_periods.op_hours_cum > found.part_start)
                                & (unit_periods.op_hours_cum <= last_visible)]
    if last_visible - found.part_start > HOURS_FOR_TREND and len(visible) >= PERIODS_FOR_TREND:
        predicted = hours_to_level(visible)
        if not np.isnan(predicted):
            rows.append({"unit_id": found.unit_id,
                         "replaced_at": found.op_hours_cum,
                         "predicted_hours": predicted,
                         "error_hours": predicted - HIDDEN_HOURS,
                         "share": HIDDEN_HOURS / predicted})
range_test = pd.DataFrame(rows)
share_low  = np.percentile(range_test.share, 10)
share_high = np.percentile(range_test.share, 90)
save_table(range_test, "range_test")

print()
print(pd.DataFrame({"parts": [len(range_test)]}, index=["rule (a) test, parts tested"]).to_string())
print()
print(pd.DataFrame({"hours left as a share of hours predicted":
                    [round(share_low, 2), round(np.median(range_test.share), 2),
                     round(share_high, 2)]},
                   index=["at 10%", "median", "at 90%"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC **The part now fitted in every unit**
# MAGIC - The part now fitted went in at the last replacement found on the unit (section 4), or at the start of the unit if none was found. The hours on the part are counted from there, not from the start of the unit.
# MAGIC - Its type comes from the label of that replacement (section 5).
# MAGIC - Early warning: the test of section 3, applied to the part now fitted.
# MAGIC - Hours remaining by rule (a) or rule (b), with a low and a high value. For (a): the hours remaining times the 10% and 90% shares. For (b): the low and high lives in the life table, minus the hours on the part.
# MAGIC - A replacement that the method has not found yet leaves the old part in this table.

# COMMAND ----------

rows = []
for unit_id in units.unit_id:
    unit_periods = periods_by_unit[unit_id]

    # the part now fitted went in at the last replacement found, or at the start of the unit
    found_here = replacements[replacements.unit_id == unit_id]
    if len(found_here) > 0:
        last_found = found_here.sort_values("op_hours_cum").iloc[-1]
        part_start = last_found.op_hours_cum
        part_type  = PART_TYPE[last_found.label]
    else:
        part_start = 0.0
        part_type  = "new install"
    part_periods  = unit_periods[unit_periods.op_hours_cum > part_start]
    hours_on_part = unit_periods.op_hours_cum.max() - part_start

    # early warning on this part, as in section 3
    warning_hours = first_warning(part_periods, EARLY_WARNING_THRESHOLD)
    has_warning   = not np.isnan(warning_hours)

    # hours remaining: rule (a) where the part has a trend that can be extended, else rule (b)
    by_trend = np.nan
    if hours_on_part > HOURS_FOR_TREND and len(part_periods) >= PERIODS_FOR_TREND:
        by_trend = hours_to_level(part_periods)

    if not np.isnan(by_trend):
        rule      = "(a) trend"
        remaining = max(by_trend, 0.0)
        low       = remaining * share_low
        high      = remaining * share_high
    else:
        rule      = "(b) life table"
        remaining = life_table.expected[part_type] - hours_on_part
        low       = life_table.low[part_type] - hours_on_part
        high      = life_table.high[part_type] - hours_on_part

    rows.append({"unit_id": unit_id, "part_type": part_type, "hours_on_part": hours_on_part,
                 "has_warning": has_warning, "rule": rule, "hours_remaining": remaining,
                 "low": low, "high": high})
fleet_now = pd.DataFrame(rows)

# the spread used in the draws below, from the width of the range
fleet_now["spread"] = (fleet_now.high - fleet_now.low) / RANGE_IN_SD
fleet_now.loc[fleet_now.spread < 60, "spread"] = 60
for column in ["hours_remaining", "low", "high"]:
    fleet_now[column] = fleet_now[column].clip(0, HORIZON_HOURS)

rows = []
for rule in ["(a) trend", "(b) life table"]:
    chosen = fleet_now[fleet_now.rule == rule]
    rows.append({"rule": rule, "units": len(chosen),
                 "hours on the part, median": int(round(chosen.hours_on_part.median())),
                 "hours remaining, median": int(round(chosen.hours_remaining.median()))})
print(pd.DataFrame(rows).to_string(index=False))

print()
print(fleet_now.part_type.value_counts().to_frame("units").to_string())

print()
print(pd.DataFrame({"units": [int(fleet_now.has_warning.sum())]},
                   index=["units with an early warning on the part now fitted"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC **From hours to quarters**
# MAGIC - Hours per quarter: each unit's own operating hours per quarter, the average of its last 4 quarters. It turns hours remaining into a calendar quarter.
# MAGIC - 300 draws: the hours remaining are an estimate, so the count is repeated 300 times. In each draw every unit's hours remaining are drawn at random around its estimate, from a normal distribution whose 10% to 90% range is as wide as the unit's low-to-high range.
# MAGIC - The table gives the average count per quarter over the 300 draws, and the 10% and 90% counts.
# MAGIC - Overdue now: the part is already at or past the replacement level.
# MAGIC - History: the failures found by the method in each past quarter, with the actual failures beside them.
# MAGIC - Simulated outcome: the simulation gives every part a wear curve, and that curve continues past the last data point. So it is known in which quarter each part now fitted reaches the replacement level. It is used only to check the forecast.
# MAGIC - This check is possible only on simulated data. In the field the check is a backtest: forecast from an earlier date and compare with the failures that followed.

# COMMAND ----------

QUARTERS_AHEAD = 6      # quarters in the forecast
QUARTERS_BACK  = 8      # quarters of history shown
DRAWS          = 300    # times the forecast is repeated

# operating hours per quarter of every unit, from its last 4 quarters
last_quarter  = periods.ts_end.max().to_period("Q")
rate_quarters = [last_quarter - 3, last_quarter - 2, last_quarter - 1, last_quarter]
hours_per_quarter = []
for unit_id in units.unit_id:
    unit_periods      = periods_by_unit[unit_id]
    quarter_of_period = unit_periods.ts_end.dt.to_period("Q")
    rates = []
    for quarter in rate_quarters:
        in_quarter = unit_periods[quarter_of_period == quarter]
        if len(in_quarter) == 0:
            continue
        days = (in_quarter.ts_end.max() - in_quarter.ts_start.min()).total_seconds() / 86400
        if days >= 20:                                   # a quarter with too few days is skipped
            rates.append(len(in_quarter) * PERIOD_HOURS / days * 91.3)
    if len(rates) > 0:
        hours_per_quarter.append(np.mean(rates))
    else:
        hours_per_quarter.append(np.nan)
fleet_now["hours_per_quarter"] = hours_per_quarter
fleet_now["hours_per_quarter"] = fleet_now.hours_per_quarter.fillna(fleet_now.hours_per_quarter.median())

# 300 draws. in each draw every unit's hours remaining are drawn around its estimate.
# quarter 0 is overdue now, quarter 1 is the first quarter ahead
counts = np.zeros((DRAWS, QUARTERS_AHEAD + 1))
for draw in range(DRAWS):
    drawn_hours = random_numbers.normal(fleet_now.hours_remaining, fleet_now.spread)
    quarter_due = np.ceil(drawn_hours / fleet_now.hours_per_quarter)
    quarter_due[drawn_hours <= 0] = 0
    for k in range(QUARTERS_AHEAD + 1):
        counts[draw, k] = (quarter_due == k).sum()

# the simulated outcome: the quarter in which the part now fitted reaches the replacement level
# in the simulation. it is used only to check the forecast
first_ahead    = last_quarter + 1
ahead_quarters = []
for k in range(QUARTERS_AHEAD):
    ahead_quarters.append(first_ahead + k)
actual_life = read_table("actual_remaining_life", ["unit_id"])
outcome_due = actual_life.due_date.dt.to_period("Q")
outcome     = [int((outcome_due < first_ahead).sum())]
for quarter in ahead_quarters:
    outcome.append(int((outcome_due == quarter).sum()))

quarter_names = ["overdue now"]
for quarter in ahead_quarters:
    quarter_names.append(str(quarter))
due = pd.DataFrame({"quarter": quarter_names,
                    "expected": counts.mean(axis=0).round(1),
                    "low": np.percentile(counts, 10, axis=0).round().astype(int),
                    "high": np.percentile(counts, 90, axis=0).round().astype(int),
                    "simulated_outcome": outcome})

# history: failures found by the method in each quarter, and the actual failures
found_quarter  = replacements[replacements.label == "A"].ts_end.dt.to_period("Q")
actual_quarter = actual[actual.actual_class == "A"].ts.dt.to_period("Q")
rows = []
for k in range(QUARTERS_BACK - 1, -1, -1):
    quarter = last_quarter - k
    rows.append({"quarter": str(quarter),
                 "found": int((found_quarter == quarter).sum()),
                 "actual": int((actual_quarter == quarter).sum())})
history = pd.DataFrame(rows)

print(history.rename(columns={"found": "failures found", "actual": "actual failures"})
             .to_string(index=False))
print()
print(due.rename(columns={"expected": "failures due, expected", "low": "10%", "high": "90%",
                          "simulated_outcome": "simulated outcome"}).to_string(index=False))

total_expected = int(round(due.expected[1:].sum()))
total_outcome  = int(due.simulated_outcome[1:].sum())
inside         = int(((due.simulated_outcome >= due.low) & (due.simulated_outcome <= due.high)).sum())
print()
print(pd.DataFrame({"count": [total_expected, total_outcome, inside, len(due)]},
                   index=["six quarters ahead, expected", "six quarters ahead, simulated outcome",
                          "rows where the simulated outcome is inside the 10% to 90% range",
                          "rows in the table"]).to_string())

# COMMAND ----------

x_back  = np.arange(len(history))
x_ahead = len(history) + np.arange(len(due))

fig, ax = plt.subplots()
ax.bar(x_back, history.found, color="0.55", label="failures found so far")
ax.bar(x_ahead, due.expected, color="C0", label="failures due, forecast")
ax.errorbar(x_ahead, due.expected, yerr=[due.expected - due.low, due.high - due.expected],
            fmt="none", ecolor="k", capsize=4, linewidth=1.5, label="10% to 90% range")
ax.plot(x_ahead, due.simulated_outcome, "D", color="C3", markersize=8, linestyle="none",
        label="check: outcome in the simulation")
ax.axvline(len(history) - 0.5, color="k", linestyle=":", linewidth=1)
ax.set_xticks(list(x_back) + list(x_ahead))
ax.set_xticklabels(list(history.quarter) + list(due.quarter), rotation=45, ha="right")
ax.set_ylabel("failures per quarter")
ax.set_title("Failures per quarter: found so far, then forecast")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("charts/forecast.png")
plt.show()

forecast_by_quarter = pd.concat([history, due], ignore_index=True)
forecast_by_quarter["row"] = np.arange(len(forecast_by_quarter))
save_table(fleet_now, "forecast_by_unit")
save_table(forecast_by_quarter, "forecast_by_quarter")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this means
# MAGIC
# MAGIC - Planning should count 24 failures, not the 15 that the service reports show. Life estimates and spare-parts numbers built on the reports would be off by that much.
# MAGIC - Life estimates need the type of each replacement, not only its date. A part fitted on its own after a failure lasts shorter than a part fitted along with other parts at scheduled work, so the life table keeps them apart.
# MAGIC - 76 units carry an early warning today. In the test the warning came about 25 weeks of operation before the replacement. That's enough time to order the part and plan the visit.
# MAGIC - Failures are expected to rise from 7 last quarter to about 23 next quarter, and about 30 a quarter in the first half of 2027. Parts and crews should be planned for that now.
# MAGIC
# MAGIC **Limits**
# MAGIC
# MAGIC - The data is simulated, so the two classes separate more cleanly than they would in the field. A real fleet has odd cases, and needs more indicators and engineered features than the four used here.
# MAGIC - Only 29 replacements were inspected, so the classifier test is small.
# MAGIC - The thresholds were set and scored on the same 38 replacements. That makes the scores look better than they would be on new data. The proper test is to fix the thresholds and score them on later replacements.
# MAGIC - The method needs 60 to 80 operating hours of data after a replacement before it can find it. Until then the newest replacement is missed, one of 38 here. That unit stays on the warning list and in the forecast with its old part, so the service report is still needed for the newest replacements.
# MAGIC - 75 of the 76 warnings are on parts that are not replaced yet, so they cannot be judged yet. In the simulated data, 9 of them are on parts still more than 2,000 operating hours from the replacement level.
# MAGIC - The forecast is better on totals than on timing: 150 failures expected over the six quarters against 160 in the simulated outcome, but 23 against 17 in the first quarter. The simulated outcome is inside the range in 4 of 7 rows.
# MAGIC - The life table used here is the one the simulation was built with. So the check against the simulated outcome tests the trend rule and the labels, not the life table.
