import numpy as np
import pandas as pd
import pytest

from coxrmstsimpson import simpson_rmst, cox_rmst


def test_simpson_rule_linear_curve():
    assert simpson_rmst([0, 1, 2, 3, 4], [1, .8, .6, .4, .2]) == pytest.approx(2.4)


def test_requires_three_points():
    with pytest.raises(ValueError):
        simpson_rmst([0, 1], [1, .5])


def test_requires_increasing_time():
    with pytest.raises(ValueError):
        simpson_rmst([0, 2, 1], [1, .5, .2])


def test_automatic_binary_treatment_profiles_and_rmst(monkeypatch):
    lifelines = pytest.importorskip("lifelines")
    from lifelines import CoxPHFitter
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(123)
    n = 120
    group = rng.binomial(1, 0.5, size=n)
    x = rng.normal(size=n)
    event_time = rng.exponential(scale=1 / (0.12 * np.exp(0.55 * group + 0.25 * x)))
    censor_time = rng.exponential(scale=12, size=n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    df = pd.DataFrame({"time": duration, "event": event, "group": group, "x": x})

    model = CoxPHFitter()
    model.fit(df, duration_col="time", event_col="event")

    monkeypatch.setattr(plt, "show", lambda: None)
    result = cox_rmst(
        model,
        treatment="group",
        tau=5,
        labels=["Control", "Treatment"],
        title="RMST by treatment",
        n_grid=101,
    )

    assert result["newdata"]["group"].tolist() == [0.0, 1.0]
    assert list(result["results"]["profile"]) == ["Control", "Treatment"]
    assert result["results"]["RMST"].notna().all()
    assert result["curves"]["profile"].nunique() == 2
    assert result["ax"].get_title() == "RMST by treatment"
    plt.close("all")
