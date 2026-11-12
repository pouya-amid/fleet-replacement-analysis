# Databricks notebook source
# MAGIC %md
# MAGIC # 02: detailed checks
# MAGIC - Reads the tables saved by `01_replacement_analysis`, so both notebooks show the same numbers.
# MAGIC - Holds the checks behind the choices in 01: the factors, the index threshold, the early-warning settings, the classifier, the remaining life and the forecast range.
# MAGIC - Actual: the data is simulated, so the actual events are known. They are used only to check results.

# COMMAND ----------

# MAGIC %pip install -q -U xgboost threadpoolctl

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCHEMA            = "workspace.pge_demo"
PERIOD_HOURS      = 20
REPLACEMENT_LEVEL = 0.50
MATCH_HOURS       = 500
CLASS_WORDS       = {"A": "failure (A)", "B": "non-failure (B)"}

# the choices made in 01
INDEX_THRESHOLD         = 2.0
EARLY_WARNING_THRESHOLD = 0.30
SMOOTH_PERIODS          = 10
LEAD_HOURS              = 400
SETTLE_HOURS            = 2000

plt.rcParams.update({"figure.figsize": (10, 5.6), "font.size": 12})

def read_table(name, order):
    table = spark.table(SCHEMA + "." + name).toPandas()
    table = table.sort_values(order).reset_index(drop=True)
    return table

factors      = read_table("period_factors", ["unit_id", "op_hours_cum"])
replacements = read_table("replacements_found", ["unit_id", "op_hours_cum"])
accounting   = read_table("replacement_accounting", ["unit_id", "op_hours"])
cases        = read_table("early_warning_cases", ["unit_id", "replaced_at"])
by_unit      = read_table("forecast_by_unit", ["unit_id"])
by_quarter   = read_table("forecast_by_quarter", ["row"])
range_test   = read_table("range_test", ["unit_id"])
reports      = read_table("service_reports", ["unit_id", "op_hours_at_service"])
actual_life  = read_table("actual_remaining_life", ["unit_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The factors: normal running against reported replacements
# MAGIC - Normal running: every period more than 500 operating hours away from any service report on the unit.
# MAGIC - At a reported replacement: for each report, the period with the highest index within 500 operating hours of it.
# MAGIC - A factor belongs in the index if it is near 0 in normal running, near 1 at a replacement, and needs little data after the replacement.
# MAGIC - Seven factors were tried. Four are in the index. Three are left out, and the last column says why.
# MAGIC   - Close to new after: 1 when the loss after is 0, falling to 0 at a loss of 0.10.
# MAGIC   - Losing before: the rate of loss before, where 0.12 per 1,000 operating hours counts as 1.
# MAGIC   - Flatter after: the rate of loss before minus the rate over the 25 periods after, on the same scale.

# COMMAND ----------

RATE_PERIODS = 25

def to_factor(values, zero_at, one_at):
    # for a whole column: 0 at or below zero_at, 1 at or above one_at, a straight line in between
    return ((values - zero_at) / (one_at - zero_at)).clip(0.0, 1.0)

# the three factors that were tried and left out
factors["factor_close_to_new"]  = 1.0 - to_factor(factors.loss_after, 0.0, 0.10)
factors["factor_losing_before"] = to_factor(factors.loss_rate_before, 0.0, 0.12)

# rate of loss after: a straight line through the 25 periods after each period
factors["loss_rate_after"] = np.nan
for unit_id in factors.unit_id.unique():
    table = factors[factors.unit_id == unit_id]
    loss  = table.loss.values
    hours = table.op_hours_cum.values.astype(float)
    rate_after = np.full(len(table), np.nan)
    for i in range(len(table) - RATE_PERIODS):
        later_hours = hours[i + 1 : i + 1 + RATE_PERIODS] - hours[i]
        later_loss  = loss[i + 1 : i + 1 + RATE_PERIODS]
        line = np.polyfit(later_hours, later_loss, 1)
        rate_after[i] = 1000 * line[0]
    factors.loc[table.index, "loss_rate_after"] = rate_after
factors["factor_flatter_after"] = to_factor(factors.loss_rate_before - factors.loss_rate_after,
                                            0.0, 0.12)

# normal running, and the period of each reported replacement
complete = factors.loss_before.notna() & factors.loss_after.notna()
factors["near_report"] = False
at_report = []
for i, report in reports.iterrows():
    near = ((factors.unit_id == report.unit_id)
            & ((factors.op_hours_cum - report.op_hours_at_service).abs() <= MATCH_HOURS))
    factors.loc[near, "near_report"] = True
    if (near & complete).any():
        at_report.append(factors[near & complete].change_index.idxmax())
normal         = factors[complete & ~factors.near_report]
at_replacement = factors.loc[at_report]

# column, name, in the index, hours of data needed after, why it was left out
tried = [["factor_step",           "size of the step",     "yes", 60,  ""],
         ["factor_scatter",        "step against scatter", "yes", 60,  ""],
         ["factor_clear",          "clear of the scatter", "yes", 60,  ""],
         ["factor_out_of_service", "out of service",       "yes", 20,  ""],
         ["factor_close_to_new",   "close to new after",   "no",  60,  "high in normal running"],
         ["factor_losing_before",  "losing before",        "no",  0,   "high in normal running"],
         ["factor_flatter_after",  "flatter after",        "no",  500, "needs 500 hours after"]]
rows = []
for item in tried:
    column = item[0]
    rows.append({"factor": item[1],
                 "in the index": item[2],
                 "normal running, median": normal[column].median(),
                 "normal running, 95%": normal[column].quantile(0.95),
                 "at reported replacements, median": at_replacement[column].median(),
                 "hours needed after": item[3],
                 "left out because": item[4]})
print(pd.DataFrame(rows).round(2).to_string(index=False))

rows = []
for name, chosen in [["normal running", normal], ["at reported replacements", at_replacement]]:
    rows.append({"": name,
                 "periods": len(chosen),
                 "index, median": round(chosen.change_index.median(), 2),
                 "index 1 or more": int((chosen.change_index >= 1).sum()),
                 "index 2 or more": int((chosen.change_index >= 2).sum()),
                 "index 3 or more": int((chosen.change_index >= 3).sum())})
print()
print(pd.DataFrame(rows).to_string(index=False))

# periods in normal running at or above the threshold: are they at an actual replacement?
high   = normal[normal.change_index >= INDEX_THRESHOLD]
nearby = 0
for i, row in high.iterrows():
    same_unit = accounting[accounting.unit_id == row.unit_id]
    if ((same_unit.op_hours - row.op_hours_cum).abs() <= 3 * PERIOD_HOURS).any():
        nearby = nearby + 1
print()
print(pd.DataFrame({"periods": [len(high), nearby, len(high) - nearby]},
                   index=["normal running, index at or above the threshold",
                          "at an actual replacement that has no report",
                          "at no actual replacement"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Loss and index on one unit
# MAGIC - One unit with a replacement found. Top: the loss of every period. Bottom: the index.
# MAGIC - Dashed line: the replacement found. Dotted lines: the replacement level and the index threshold.

# COMMAND ----------

unit_id = replacements.unit_id.value_counts().index[0]
table   = factors[factors.unit_id == unit_id]
found   = replacements[replacements.unit_id == unit_id]

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
top.plot(table.op_hours_cum, table.loss, linewidth=1)
top.axhline(REPLACEMENT_LEVEL, color="C3", linestyle=":", linewidth=1.5)
top.set_ylabel("loss")
top.set_title("Unit " + unit_id)
bottom.plot(table.op_hours_cum, table.change_index, color="C1", linewidth=1)
bottom.axhline(INDEX_THRESHOLD, color="k", linestyle=":", linewidth=1.5)
bottom.set_ylabel("index (0 to 4)")
bottom.set_xlabel("operating hours on the unit")
for hours in found.op_hours_cum:
    top.axvline(hours, color="k", linestyle="--", linewidth=1)
    bottom.axvline(hours, color="k", linestyle="--", linewidth=1)
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Dates, and how soon a replacement is found
# MAGIC - Date error: operating hours between the actual replacement and the date from the method, and the same for the service report.
# MAGIC - The method dates a replacement at the end of the period that holds the step, so its date runs up to 20 hours late.
# MAGIC - How soon: actual replacements grouped by the hours of record after them, found and not found.

# COMMAND ----------

found_error  = (accounting.found_at_hours - accounting.op_hours).abs().dropna()
report_error = (accounting.report_hours - accounting.op_hours).abs().dropna()
dates = pd.DataFrame({"replacements":            [len(found_error), len(report_error)],
                      "median error, hours":     [found_error.median(), report_error.median()],
                      "worst error, hours":      [found_error.max(), report_error.max()],
                      "more than 100 hours off": [int((found_error > 100).sum()),
                                                  int((report_error > 100).sum())]},
                     index=["date from the method", "date in the service report"])
print(dates.round(0).astype(int).to_string())

bands = pd.cut(accounting.hours_after, [-1, 80, 200, 500, 100000],
               labels=["under 80", "80 to 200", "200 to 500", "over 500"])
found_words = np.where(accounting.found, "found", "not found")
how_soon = pd.crosstab(bands, found_words)
how_soon.index.name   = "hours of record after the replacement"
how_soon.columns.name = None
print()
print(how_soon.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Early-warning settings
# MAGIC - The test from section 3 of 01, repeated for six thresholds and for five averaging lengths.
# MAGIC - Failures caught and missed, and the median lead time of those caught.
# MAGIC - Non-failures warned early: a warning before a scheduled change. It counts as neither a catch nor a false alarm.
# MAGIC - Never replaced, warned: units with a warning on a part that was never replaced.
# MAGIC   - False alarms: of these, the unit ran 2,000 more operating hours after the warning.
# MAGIC   - Too early (actual): of these, the part was actually more than 2,000 operating hours from the replacement level when the warning came. This uses the wear curve of the simulation, so it is known only here.
# MAGIC - Reading the tables: a lower threshold or a shorter average warns earlier, but more of the warnings are too early. A higher threshold or a longer average misses failures.
# MAGIC - Of the settings tested, 0.30 with 10 periods is the one that catches all 24 failures with the fewest warnings that are too early.

# COMMAND ----------

def average_of_last_periods(loss, smooth_periods):
    # for every period: the average loss of that period and the periods before it. backward only.
    average = np.full(len(loss), np.nan)
    for i in range(smooth_periods - 1, len(loss)):
        average[i] = loss[i - smooth_periods + 1 : i + 1].mean()
    return average

def first_warning(part_periods, threshold, smooth_periods):
    # operating hours at which the average first reaches the threshold. NaN if it never does.
    average = average_of_last_periods(part_periods.loss.values, smooth_periods)
    hours   = part_periods.op_hours_cum.values
    for i in range(len(average)):
        if average[i] >= threshold:
            return float(hours[i])
    return np.nan

periods_by_unit = {}
for unit_id in factors.unit_id.unique():
    periods_by_unit[unit_id] = factors[factors.unit_id == unit_id]
life_by_unit = actual_life.set_index("unit_id")

def run_test(threshold, smooth_periods):
    # the test of section 3 in 01, for one threshold and one averaging length
    caught       = 0
    missed       = 0
    early_change = 0      # non-failures with a warning before the scheduled change
    warned       = 0      # never-replaced units with a warning
    false_alarms = 0      # of these, ran 2,000 hours after the warning
    too_early    = 0      # of these, more than 2,000 hours from the replacement level (actual)
    leads        = []
    for i, case in cases.iterrows():
        unit_periods = periods_by_unit[case.unit_id]
        if case.kind == "never replaced":
            part_periods = unit_periods
        else:
            last_hours   = case.replaced_at - LEAD_HOURS
            part_periods = unit_periods[(unit_periods.op_hours_cum > case.part_start)
                                        & (unit_periods.op_hours_cum <= last_hours)]
        warning_hours = first_warning(part_periods, threshold, smooth_periods)
        has_warning   = not np.isnan(warning_hours)

        if case.kind == "failure":
            if has_warning:
                caught = caught + 1
                leads.append(case.replaced_at - warning_hours)
            else:
                missed = missed + 1
        elif case.kind == "non-failure":
            if has_warning:
                early_change = early_change + 1
        elif has_warning:
            warned = warned + 1
            if unit_periods.op_hours_cum.max() - warning_hours >= SETTLE_HOURS:
                false_alarms = false_alarms + 1
            life = life_by_unit.loc[case.unit_id]
            hours_to_level = life.hours_on_unit - warning_hours + life.hours_remaining
            if hours_to_level > 2000:
                too_early = too_early + 1

    if len(leads) > 0:
        median_lead = int(round(np.median(leads)))
    else:
        median_lead = 0
    return {"failures caught": caught,
            "failures missed": missed,
            "median lead, hours": median_lead,
            "non-failures warned early": early_change,
            "never replaced, warned": warned,
            "of these, false alarms": false_alarms,
            "of these, too early (actual)": too_early}

rows = []
for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    row = {"threshold": threshold}
    row.update(run_test(threshold, SMOOTH_PERIODS))
    rows.append(row)
print("Averaging length " + str(SMOOTH_PERIODS) + " periods, by threshold")
print(pd.DataFrame(rows).to_string(index=False))

rows = []
for smooth_periods in [1, 5, 10, 25, 50]:
    row = {"periods averaged": smooth_periods, "operating hours": smooth_periods * PERIOD_HOURS}
    row.update(run_test(EARLY_WARNING_THRESHOLD, smooth_periods))
    rows.append(row)
print()
print("Threshold " + str(EARLY_WARNING_THRESHOLD) + ", by averaging length")
print(pd.DataFrame(rows).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The classifier
# MAGIC - Every replacement found, by the label from the method and by the actual class. This includes the ones that were not inspected.
# MAGIC - The two K-means groups against the actual class.
# MAGIC - Every 5-fold test label that disagrees with the inspection, with its measurements and a plot.

# COMMAND ----------

label_words    = "labelled " + replacements.label.map(CLASS_WORDS)
actual_words   = replacements.actual_class.map(CLASS_WORDS).fillna("no actual replacement")
inspected_or_not = np.where(replacements.inspected_class == "", "not inspected", "inspected")

against_actual = pd.crosstab([inspected_or_not, label_words], actual_words)
against_actual.index.names  = [None, None]
against_actual.columns.name = None
print(against_actual.to_string())

groups = pd.crosstab(replacements.group, actual_words)
groups.index.name   = None
groups.columns.name = None
print()
print(groups.to_string())

tested = replacements[replacements.test_label != ""]
misses = tested[tested.test_label != tested.inspected_class]
shown  = misses[["unit_id", "op_hours_cum", "inspected_class", "test_label", "actual_class",
                 "loss_before", "step", "step_vs_scatter", "loss_rate_before", "hours_on_part"]]
shown.columns = ["unit", "operating hours", "inspected", "test label", "actual", "loss before",
                 "step", "step against scatter", "rate of loss before", "hours on the part"]
print()
print(pd.DataFrame({"count": [len(tested), len(misses)]},
                   index=["inspected replacements in the 5-fold test",
                          "test label disagrees with the inspection"]).to_string())
if len(misses) > 0:
    print()
    print(shown.round(2).to_string(index=False))

# COMMAND ----------

for i, miss in misses.iterrows():
    table = factors[factors.unit_id == miss.unit_id]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table.op_hours_cum, table.loss, linewidth=1)
    ax.axvline(miss.op_hours_cum, color="k", linestyle="--", linewidth=1)
    ax.axhline(REPLACEMENT_LEVEL, color="C3", linestyle=":", linewidth=1.5)
    ax.set_title(miss.unit_id + ": inspected " + miss.inspected_class + ", test label "
                 + miss.test_label)
    ax.set_xlabel("operating hours on the unit")
    ax.set_ylabel("loss")
    fig.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Is the classifier overfit?**
# MAGIC - An overfit model is right on the parts it was trained on and wrong on parts it has not seen.
# MAGIC - Check 1: the model trained on all inspected parts and tested on the same parts, against the 5-fold test, where every part is labelled by a model that did not see it. A large gap between the two means overfit.
# MAGIC - Check 2: the 5-fold test repeated with 20 different random splits. The result should not depend on the split.
# MAGIC - Check 3: the same 20 splits with other models: XGBoost with its default settings, a very small XGBoost of 10 trees, and a simple rule on one measurement. The rule calls a part a failure if its loss before is above the midpoint between the highest non-failure and the lowest failure in the training parts. If these do as well, the result does not come from the settings or the size of the model.
# MAGIC - Check 4: the replacements that were not inspected. The model was never trained on them. Their actual class is known here because the data is simulated.

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

MEASUREMENTS = ["loss_before", "step", "step_vs_scatter", "loss_rate_before", "hours_on_part"]
SPLITS       = 20      # number of different random splits

inspected   = replacements[replacements.inspected_class != ""]
inputs      = inspected[MEASUREMENTS].values
is_failure  = (inspected.inspected_class == "A").astype(int).values
loss_before = inspected.loss_before.values

def model_as_in_01():
    return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                         eval_metric="logloss", random_state=0)

def model_with_default_settings():
    return XGBClassifier(eval_metric="logloss", random_state=0)

def very_small_model():
    return XGBClassifier(n_estimators=10, max_depth=1, eval_metric="logloss", random_state=0)

def five_fold_correct(make_model, split_number):
    # how many inspected parts are labelled correctly by a model that did not see them
    folds   = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_number)
    correct = 0
    for train_rows, test_rows in folds.split(inputs, is_failure):
        model = make_model()
        model.fit(inputs[train_rows], is_failure[train_rows])
        predicted = model.predict(inputs[test_rows])
        correct   = correct + int((predicted == is_failure[test_rows]).sum())
    return correct

def five_fold_correct_simple_rule(split_number):
    # the same test for the simple rule on the loss before
    folds   = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_number)
    correct = 0
    for train_rows, test_rows in folds.split(inputs, is_failure):
        train_loss          = loss_before[train_rows]
        train_failure       = is_failure[train_rows]
        highest_non_failure = train_loss[train_failure == 0].max()
        lowest_failure      = train_loss[train_failure == 1].min()
        threshold = (highest_non_failure + lowest_failure) / 2
        predicted = (loss_before[test_rows] >= threshold).astype(int)
        correct   = correct + int((predicted == is_failure[test_rows]).sum())
    return correct

# check 1: tested on the parts it was trained on, against parts it did not see
model = model_as_in_01()
model.fit(inputs, is_failure)
on_training = int((model.predict(inputs) == is_failure).sum())
not_seen    = five_fold_correct(model_as_in_01, 0)
print(pd.DataFrame({"correct": [on_training, not_seen], "out of": [len(inspected), len(inspected)]},
                   index=["tested on the parts it was trained on",
                          "5-fold test, parts it did not see"]).to_string())

# checks 2 and 3: 20 different splits, four models
rows = []
for name, make_model in [["XGBoost as in 01", model_as_in_01],
                         ["XGBoost, default settings", model_with_default_settings],
                         ["XGBoost, 10 trees of depth 1", very_small_model]]:
    results = []
    for split_number in range(SPLITS):
        results.append(five_fold_correct(make_model, split_number))
    rows.append({"model": name, "lowest": min(results), "median": int(np.median(results)),
                 "highest": max(results), "out of": len(inspected)})
results = []
for split_number in range(SPLITS):
    results.append(five_fold_correct_simple_rule(split_number))
rows.append({"model": "simple rule on the loss before", "lowest": min(results),
             "median": int(np.median(results)), "highest": max(results), "out of": len(inspected)})
print()
print("5-fold test, parts labelled correctly, over " + str(SPLITS) + " different splits")
print(pd.DataFrame(rows).to_string(index=False))

# check 4: the replacements that were not inspected
not_inspected = replacements[(replacements.inspected_class == "") & (replacements.actual_class != "")]
correct       = int((not_inspected.label == not_inspected.actual_class).sum())
print()
print(pd.DataFrame({"correct": [correct], "out of": [len(not_inspected)]},
                   index=["not inspected, label against the actual class"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Remaining life
# MAGIC - Each unit's hours remaining from 01, against the actual hours until the part now fitted reaches the replacement level. Both are held between 0 and 6,000.
# MAGIC - By rule: the size of the error, and how often the actual value is inside the range.
# MAGIC - The test behind the range of rule (a): the hours left as a share of the hours predicted, and the error in hours, each at 10%, 50% and 90%.

# COMMAND ----------

check = by_unit.merge(actual_life[["unit_id", "hours_remaining"]], on="unit_id",
                      suffixes=("", "_actual"))
check["actual_hours"] = check.hours_remaining_actual.clip(0, 6000)
check["error"]        = (check.hours_remaining - check.actual_hours).abs()
check["inside"]       = (check.actual_hours >= check.low) & (check.actual_hours <= check.high)

rows = []
for name, chosen in [["(a) trend", check[check.rule == "(a) trend"]],
                     ["(b) life table", check[check.rule == "(b) life table"]],
                     ["all units", check]]:
    rows.append({"rule": name,
                 "units": len(chosen),
                 "mean error, hours": round(chosen.error.mean()),
                 "median error, hours": round(chosen.error.median()),
                 "actual inside the range, share": round(chosen.inside.mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print()
print(pd.DataFrame({"hours left as a share of hours predicted":
                    np.percentile(range_test.share, [10, 50, 90]).round(2),
                    "predicted minus the 1,000 hours left, hours":
                    np.percentile(range_test.error_hours, [10, 50, 90]).round(0)},
                   index=["at 10%", "at 50%", "at 90%"]).to_string())
print()
print(pd.DataFrame({"parts": [len(range_test)]}, index=["rule (a) test, parts tested"]).to_string())

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
chosen = check[check.rule == "(a) trend"]
left.scatter(chosen.actual_hours, chosen.hours_remaining, s=18, marker="o", alpha=0.6,
             label="(a) trend")
chosen = check[check.rule == "(b) life table"]
left.scatter(chosen.actual_hours, chosen.hours_remaining, s=18, marker="s", alpha=0.6,
             label="(b) life table")
left.plot([0, 6000], [0, 6000], color="k", linestyle=":", linewidth=1)
left.set_xlabel("actual hours remaining")
left.set_ylabel("hours remaining from 01")
left.set_title("Remaining life by rule")
left.legend()
right.hist(range_test.error_hours, bins=12, color="0.6")
right.axvline(0, color="k", linewidth=1)
right.set_xlabel("predicted minus the 1,000 hours left")
right.set_ylabel("parts")
right.set_title("Rule (a) test: error")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. The forecast against the simulated outcome
# MAGIC - Simulated outcome: the quarter in which each part now fitted reaches the replacement level in the simulation. The simulation holds the wear of every part beyond the last data point, so this is known here. It would not be known in the field.
# MAGIC - For each quarter ahead: whether the simulated outcome is inside the 10% to 90% range.
# MAGIC - Units with an actual replacement in the last 200 hours of the record: whether the method found it, and the quarter the forecast puts the unit in. Quarter 0 is overdue now, quarter 1 is the first quarter ahead. A replacement that is not found yet leaves the old part in the forecast.

# COMMAND ----------

ahead = by_quarter[by_quarter.expected.notna()].copy()
for column in ["low", "high", "simulated_outcome"]:
    ahead[column] = ahead[column].astype(int)
in_range = (ahead.simulated_outcome >= ahead.low) & (ahead.simulated_outcome <= ahead.high)
ahead["inside the range"] = np.where(in_range, "yes", "no")
shown = ahead[["quarter", "expected", "low", "high", "simulated_outcome", "inside the range"]]
shown.columns = ["quarter", "failures due, expected", "10%", "90%", "simulated outcome",
                 "inside the range"]
print(shown.to_string(index=False))

recent = accounting[accounting.hours_after <= 200]
recent = recent.merge(by_unit[["unit_id", "rule", "hours_remaining", "hours_per_quarter"]],
                      on="unit_id")
recent["quarter_due"] = np.ceil(recent.hours_remaining / recent.hours_per_quarter).astype(int)
recent["was_found"]   = np.where(recent.found, "yes", "no")
shown = recent[["unit_id", "hours_after", "was_found", "rule", "hours_remaining", "quarter_due"]]
shown.columns = ["unit", "hours of record after", "found by the method", "rule",
                 "hours remaining", "quarter due"]
print()
print("Units with an actual replacement in the last 200 hours of the record")
print(shown.round(0).to_string(index=False))
