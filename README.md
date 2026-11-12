# Separating failures from routine replacements in fleet data

Pouya Amid, November 2026

These notebooks show how I separate failure replacements from routine ones when the service
reports don't say which is which, and how that feeds a forecast of failures per quarter.

The original work was done for my employer and is confidential. I rebuilt the approach here on
simulated data for 250 units. There is no company data, code or result in this repository.

## Read the notebooks

Each page shows the code and the results as they ran.

- [Replacement analysis](01_replacement_analysis.html): the health factor, the early warning,
  finding the replacements, failure or non-failure, and the forecast.
- [Detailed checks](02_detailed_checks.html): the checks behind the thresholds, the classifier
  and the forecast range.
- [Simulation](00_simulate.html): how the simulated fleet data is made.

## Run them yourself

The notebooks run in Databricks. The free edition is enough.

1. In Databricks, open Workspace, click Create, then Git folder. Paste the address of this
   repository: `https://github.com/pouya-amid/fleet-replacement-analysis`
2. Open `00_simulate` and click Run all. It creates the simulated data, six tables.
3. Open `01_replacement_analysis` and click Run all.
4. Open `02_detailed_checks` and click Run all.

Run them in this order. Each notebook reads the tables written by the one before it. The three
take a few minutes in total.

The simulation starts from a fixed seed, so the results should match the pages above.

The tables go into one schema. Its name is set near the top of each notebook. Change it there if
your workspace has no catalog named `workspace`.

## Files

| File | What it is |
|---|---|
| `00_simulate.py`, `01_replacement_analysis.py`, `02_detailed_checks.py` | The notebooks, to run in Databricks |
| The three `.html` files | The same notebooks with their results, for reading |
