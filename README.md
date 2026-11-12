# Separating failures from routine replacements in fleet data

Pouya Amid, November 2026

These notebooks show how I separate failure replacements from routine ones when the service
reports don't say which is which, and how that feeds a forecast of failures per quarter.

The original work was done for my employer and is confidential. I rebuilt the approach here on
simulated data for 250 units. There is no company data, code or result in this repository.

## Notebooks

- [Replacement analysis](01_replacement_analysis.html): the health factor, the early warning,
  finding the replacements, failure or non-failure, and the forecast.
- [Detailed checks](02_detailed_checks.html): the checks behind the thresholds, the classifier
  and the forecast range.

The notebooks were run in Databricks. Each page shows the code and the results as they ran.
