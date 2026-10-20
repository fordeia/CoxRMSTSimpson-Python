# CoxRMSTSimpson (Python)

A Python companion to the R package [CoxRMSTSimpson](https://github.com/fordeia/CoxRMSTSimpson).

`coxrmstsimpson` estimates **Restricted Mean Survival Time (RMST)** from fitted Cox proportional hazards models using composite Simpson's rule. It predicts survival curves from a fitted `lifelines.CoxPHFitter`, estimates RMST up to a user-specified restriction time, and produces survival plots with shaded RMST areas.

## Features

- Estimate RMST from fitted `lifelines.CoxPHFitter` models.
- Compare the two levels of a binary treatment variable.
- Specify custom labels for treatment groups.
- Predict survival curves and estimate RMST up to a chosen time.
- Automatically display RMST results and a survival plot.
- Integrate survival curves using composite Simpson's rule.

## Installation

Install the package from PyPI:

```bash
python -m pip install coxrmstsimpson
```

Python 3.10 or later is required. Dependencies, including NumPy, pandas, SciPy, Matplotlib, and lifelines, are installed automatically by pip.

## Quick Start: Control versus Treatment

This example simulates survival data, fits a Cox proportional hazards model, estimates RMST up to time 10, and produces a plot comparing Control and Treatment.

```python
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from coxrmstsimpson import cox_rmst

# Simulate reproducible survival data
np.random.seed(123)

n = 200
group = np.random.binomial(1, 0.5, n)

event_time = np.random.exponential(
    scale=1 / (0.10 * np.exp(-0.5 * group))
)
censor_time = np.random.exponential(scale=15, size=n)

df = pd.DataFrame({
    "time": np.minimum(event_time, censor_time),
    "event": (event_time <= censor_time).astype(int),
    "group": group
})

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col="time", event_col="event")

# Estimate RMST and create the plot
result = cox_rmst(
    cph,
    treatment="group",
    tau=10,
    labels=["Control", "Treatment"],
    title="Restricted Mean Survival Time by Treatment"
)

# Display the RMST results
print(result["results"])
```

In this example:

- `group = 0` represents Control.
- `group = 1` represents Treatment.
- `event = 1` indicates that the event occurred.
- `event = 0` indicates a right-censored observation.
- `tau = 10` sets the restriction time for the RMST calculation.

The package predicts survival curves for the two treatment groups, estimates their RMST values, displays the results, and creates the plot. A separate plotting command or `plt.show()` call is not required for this workflow.

For your own data, fit a Cox model using the appropriate survival-time, event-status, and covariate variables. The treatment variable supplied to `cox_rmst()` must be a binary covariate included in the fitted model, coded 0 and 1.

## Using Explicit Covariate Profiles

You can also specify the covariate profiles directly using `newdata`.

For example, if the fitted Cox model includes a binary covariate named `group`, you can compare the following profiles:

```python
profiles = pd.DataFrame({
    "group": [0, 1]
})

result = cox_rmst(
    cph,
    newdata=profiles,
    tau=10
)

print(result["results"])
```

Each row of `profiles` specifies one covariate profile for which a predicted survival curve and RMST estimate are calculated. The profile columns must match the covariates and encodings used to fit the Cox model.

## Available Functions

### `cox_rmst()`

Estimates restricted mean survival time from a fitted Cox proportional hazards model.

The function supports two workflows:

**Automatic binary treatment comparison**

```python
result = cox_rmst(
    cph,
    treatment="group",
    tau=10,
    labels=["Control", "Treatment"]
)
```

**Explicit covariate profiles**

```python
result = cox_rmst(
    cph,
    newdata=profiles,
    tau=10,
    n_grid=101
)
```

The `n_grid` parameter controls the number of time-grid points used for numerical integration and must be an odd integer of at least 3.

### `plot_cox_rmst()`

Plots predicted survival curves and shades the areas beneath them up to the restriction time.

```python
from coxrmstsimpson import plot_cox_rmst

plot_cox_rmst(result)
```

This function is available when you want to plot an existing result separately. In the automatic treatment-comparison workflow, `cox_rmst()` creates the plot for you.

### `simpson_rmst()`

Estimates the area under a supplied survival curve using composite Simpson's rule.

```python
from coxrmstsimpson import simpson_rmst

rmst = simpson_rmst(time, survival)
```

Here, `time` contains the time-grid values and `survival` contains the corresponding survival probabilities.

## Mathematical Definition

Restricted mean survival time up to time \(\tau\) is the area under the survival function from time 0 to \(\tau\):

\[
\operatorname{RMST}(\tau)=\int_0^\tau S(t)\,dt.
\]

Here, \(S(t)\) is the predicted survival probability at time \(t\).

RMST provides an interpretable summary of the expected time alive over the specified follow-up interval, based on the survival curve.

## Numerical Note

`lifelines` interpolates predicted survival probabilities at requested times that are not already on its prediction timeline. This can differ from the step-function evaluation commonly used by R's `survival::survfit`.

Consequently, results from the Python and R implementations may differ numerically. Cross-language validation is recommended before claiming exact numerical equivalence.

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- SciPy
- Matplotlib
- lifelines

These dependencies are installed automatically when the package is installed using pip.

## Project Links

- **PyPI package:** https://pypi.org/project/coxrmstsimpson/
- **GitHub Python repository:** https://github.com/fordeia/CoxRMSTSimpson-Python
- **Companion R package:** https://github.com/fordeia/CoxRMSTSimpson

## Citation

If you use `coxrmstsimpson` in research, cite the software and provide the package URL. For the companion R implementation, consult its GitHub repository for the available citation information.