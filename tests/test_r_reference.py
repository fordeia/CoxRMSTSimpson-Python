
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from coxrmstsimpson import cox_rmst


def test_r_reference_rmst_values():
    """Check RMST estimates against the R package reference results."""

    data = pd.read_csv("shared_survival_data.csv")

    model = CoxPHFitter()
    model.fit(data, duration_col="time", event_col="status", formula="X")

    profiles = pd.DataFrame(
        {"X": [0, 1]},
        index=["Control", "Treatment"],
    )

    result = cox_rmst(
        model=model,
        newdata=profiles,
        tau=10,
        n_grid=101,
    )

    estimates = result["results"].set_index("profile")["RMST"]

    # Reference values from CoxRMSTSimpson in R.
    expected_control = 8.816191
    expected_treatment = 7.457006

    # Allow for small numerical differences between R and Python.
    assert np.isclose(
        estimates["Control"], expected_control, atol=0.03, rtol=0
    )
    assert np.isclose(
        estimates["Treatment"], expected_treatment, atol=0.03, rtol=0
    )
