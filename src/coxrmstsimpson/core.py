"""Core restricted mean survival time calculations."""
import numpy as np
import pandas as pd
from scipy.integrate import simpson


def simpson_rmst(time, survival) -> float:
    """Integrate survival probabilities over time using Simpson's rule."""
    t = np.asarray(time, dtype=float)
    s = np.asarray(survival, dtype=float)

    if t.ndim != 1 or s.ndim != 1 or len(t) != len(s):
        raise ValueError("time and survival must be 1D arrays of equal length.")
    if len(t) < 3:
        raise ValueError("At least three time points are required.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(s)):
        raise ValueError("time and survival must contain only finite values.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing.")
    if np.any((s < 0) | (s > 1)):
        raise ValueError("survival probabilities must lie between 0 and 1.")

    return float(simpson(s, x=t))


def _validate_model(model):
    """Validate that model is a fitted lifelines CoxPHFitter."""
    from lifelines import CoxPHFitter

    if not isinstance(model, CoxPHFitter):
        raise TypeError("model must be a fitted lifelines.CoxPHFitter.")
    if not hasattr(model, "params_"):
        raise ValueError("model must be fitted before calling cox_rmst().")


def _mean_profile(model, covariates):
    """Return fitted covariate means for prediction."""
    means = getattr(model, "_norm_mean", None)
    profile = means.to_dict() if means is not None else {}

    for name in covariates:
        if name not in profile or not np.isfinite(profile[name]):
            profile[name] = 0.0

    return profile


def cox_rmst(
    model,
    newdata: pd.DataFrame = None,
    tau: float = None,
    n_grid: int = 101,
    *,
    treatment: str = None,
    labels=None,
    title: str = None,
    show_plot: bool = True,
    show_rmst: bool = True,
    print_rmst: bool = True,
) -> dict:
    """Calculate RMST and optionally print results and display one plot.

    Automatic binary-covariate mode:
        cox_rmst(
            model,
            treatment="group",
            tau=10,
            labels=["Control", "Treatment"]
        )

    Explicit-profile mode:
        cox_rmst(
            model,
            newdata=profiles,
            tau=10,
            labels=["Control", "Treatment"]
        )

    In automatic mode, treatment names a numeric 0/1 covariate in the fitted
    model. Other covariates are held at their fitted normalization means.
    """
    _validate_model(model)

    if (
        tau is None
        or not np.isscalar(tau)
        or not np.isfinite(tau)
        or tau <= 0
    ):
        raise ValueError("tau must be a positive finite number.")

    if (
        not isinstance(n_grid, (int, np.integer))
        or isinstance(n_grid, bool)
        or n_grid < 3
        or n_grid % 2 == 0
    ):
        raise ValueError("n_grid must be an odd integer >= 3.")

    covariates = list(model.params_.index)

    if treatment is not None:
        if newdata is not None:
            raise ValueError("Provide either treatment or newdata, not both.")

        if not isinstance(treatment, str) or treatment not in covariates:
            raise ValueError(
                "treatment must name a covariate included in the fitted model: "
                f"{covariates}"
            )

        profile = _mean_profile(model, covariates)
        profile0 = dict(profile)
        profile1 = dict(profile)
        profile0[treatment] = 0
        profile1[treatment] = 1

        newdata = pd.DataFrame(
            [profile0, profile1],
            columns=covariates,
        )

        if labels is None:
            labels = ["Group 0", "Group 1"]
        else:
            labels = list(labels)
            if len(labels) != 2:
                raise ValueError("Provide exactly two labels.")

        newdata.index = [str(label) for label in labels]

    else:
        if not isinstance(newdata, pd.DataFrame) or newdata.empty:
            raise ValueError(
                "Provide newdata or specify a binary treatment covariate."
            )

        missing = set(covariates) - set(newdata.columns)
        if missing:
            raise ValueError(
                f"newdata is missing covariates: {sorted(missing)}"
            )

        newdata = newdata.loc[:, covariates].copy()

        if labels is not None:
            labels = list(labels)
            if len(labels) != len(newdata):
                raise ValueError(
                    f"Provide one label per profile ({len(newdata)} required)."
                )
            newdata.index = [str(label) for label in labels]

    grid = np.linspace(0.0, float(tau), int(n_grid))
    predicted = model.predict_survival_function(newdata, times=grid)

    profile_labels = [
        str(label) if str(label).strip() else f"Profile {i + 1}"
        for i, label in enumerate(newdata.index)
    ]

    curves = []
    rmst_values = []

    for i, label in enumerate(profile_labels):
        survival = predicted.iloc[:, i].to_numpy(dtype=float)

        curves.append(
            pd.DataFrame(
                {
                    "time": grid,
                    "survival": survival,
                    "profile": label,
                }
            )
        )

        rmst_values.append(simpson_rmst(grid, survival))

    curve_df = pd.concat(curves, ignore_index=True)

    results = pd.DataFrame(
        {
            "profile": profile_labels,
            "tau": float(tau),
            "RMST": rmst_values,
        }
    )

    result = {
        "model": model,
        "newdata": newdata.copy(),
        "tau": float(tau),
        "time_grid": grid,
        "curves": curve_df,
        "results": results,
        "treatment": treatment,
    }

    # Print the RMST table once, before opening the plot.
    if print_rmst:
        print("\nRestricted Mean Survival Time (RMST):")
        print(results.to_string(index=False))

    # Create one plot using the supplied labels.
    if show_plot:
        from .plotting import plot_cox_rmst

        ax = plot_cox_rmst(
            result,
            labels=profile_labels,
            title=title,
            show_rmst=show_rmst,
            show=True,
        )

        result["ax"] = ax
        result["figure"] = ax.figure

    return result
