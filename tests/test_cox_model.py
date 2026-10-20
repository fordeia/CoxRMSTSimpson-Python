import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from coxrmstsimpson import cox_rmst, plot_cox_rmst


def test_rmst_from_fitted_cox_model():
    rng = np.random.default_rng(123)
    n = 200

    age = rng.normal(60, 10, n)
    treatment = rng.binomial(1, 0.5, n)

    # Simulate event times from a proportional hazards model.
    hazard = 0.04 * np.exp(0.03 * (age - 60) - 0.4 * treatment)
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.exponential(30, n)

    df = pd.DataFrame({
        "time": np.minimum(event_time, censor_time),
        "status": (event_time <= censor_time).astype(int),
        "age": age,
        "treatment": treatment,
    })

    model = CoxPHFitter()
    model.fit(df, duration_col="time", event_col="status")

    profiles = pd.DataFrame(
        {"age": [60, 60], "treatment": [0, 1]},
        index=["Control", "Treatment"],
    )

    result = cox_rmst(model, profiles, tau=10, n_grid=101)

    assert len(result["results"]) == 2
    assert set(result["results"]["profile"]) == {"Control", "Treatment"}
    assert np.all(result["results"]["RMST"] > 0)
    assert np.all(result["results"]["RMST"] <= 10)
    assert len(result["curves"]) == 202
    assert result["curves"]["survival"].between(0, 1).all()

    ax = plot_cox_rmst(result)
    assert ax.get_xlabel() == "Time"
