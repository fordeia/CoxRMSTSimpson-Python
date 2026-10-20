"""Core restricted mean survival time calculations."""
import numpy as np
import pandas as pd
from scipy.integrate import simpson


def simpson_rmst(time, survival) -> float:
    """Integrate survival probabilities over time using composite Simpson's rule."""
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
    """Return the model's fitted covariate means for prediction."""
    means = getattr(model, "_norm_mean", None)
    if means is not None:
        profile = means.to_dict()
    else:
        profile = {}
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
) -> dict:
    """Estimate RMST from a fitted lifelines CoxPHFitter model.

    Two usage patterns are supported.

    Automatic binary-treatment mode:
        cox_rmst(model, treatment="group", tau=10,
                 labels=["Control", "Treatment"], title="RMST by group")

    Existing explicit-profile mode:
        cox_rmst(model, newdata, tau=10)

    In automatic mode, ``treatment`` must be a numeric 0/1 covariate included
    in the fitted model. Other model covariates are set to their fitted means
    (lifelines' model normalization means). The package plots the curves by
    default. Set ``show_plot=False`` to calculate without displaying a plot.

    Parameters
    ----------
    model : lifelines.CoxPHFitter
        A fitted Cox proportional hazards model.
    newdata : pandas.DataFrame, optional
        Explicit covariate profiles for the existing API.
    tau : float
        Positive restriction time.
    n_grid : int, default 101
        Odd number of time-grid points, at least 3.
    treatment : str, optional
        Name of a numeric 0/1 covariate in the fitted model. When supplied,
        profiles for treatment=0 and treatment=1 are created automatically.
    labels : sequence of str, optional
        Two legend/result labels in automatic mode, or one label per row of
        ``newdata`` in explicit-profile mode.
    title : str, optional
        Custom plot title.
    show_plot : bool, default True
        Display the plot before returning.
    show_rmst : bool, default True
        Include RMST values in the plot legend.

    Returns
    -------
    dict
        Includes ``curves`` (predicted survival curves), ``results`` (RMST
        table), ``newdata``, ``tau``, and model metadata.
    """
    _validate_model(model)

    if tau is None or not np.isscalar(tau) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number.")
    if not isinstance(n_grid, (int, np.integer)) or isinstance(n_grid, bool) or n_grid < 3 or n_grid % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 3.")

    covariates = list(model.params_.index)

    if treatment is not None:
        if newdata is not None:
            raise ValueError("Provide either treatment= or newdata, not both.")
        if not isinstance(treatment, str) or treatment not in covariates:
            raise ValueError(
                "treatment must name a covariate included in the fitted model "
                f"({covariates}). For automatic mode, use a numeric 0/1 covariate."
            )
        profile = _mean_profile(model, covariates)
        try:
            profile0 = dict(profile)
            profile1 = dict(profile)
            profile0[treatment] = 0.0
            profile1[treatment] = 1.0
            newdata = pd.DataFrame([profile0, profile1], columns=covariates)
        except Exception as exc:
            raise ValueError("Could not construct the treatment profiles.") from exc

        if labels is None:
            labels = ["Control", "Treatment"]
        else:
            labels = list(labels)
            if len(labels) != 2:
                raise ValueError("labels must contain exactly two labels.")
        newdata.index = [str(labels[0]), str(labels[1])]
    else:
        if not isinstance(newdata, pd.DataFrame) or newdata.empty:
            raise ValueError(
                "Provide a nonempty newdata DataFrame or use treatment='binary_covariate'."
            )
        missing = set(covariates) - set(newdata.columns)
        if missing:
            raise ValueError(f"newdata is missing covariates: {sorted(missing)}")
        newdata = newdata.loc[:, covariates].copy()
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(newdata):
                raise ValueError(
                    f"Provide one custom label for each profile ({len(newdata)} required)."
                )
            newdata.index = [str(label) for label in labels]

    grid = np.linspace(0.0, float(tau), int(n_grid))
    predicted = model.predict_survival_function(newdata, times=grid)
    profile_labels = [str(x) if str(x).strip() else f"Profile {i+1}"
                      for i, x in enumerate(newdata.index)]
    curves, values = [], []
    for i, label in enumerate(profile_labels):
        surv = predicted.iloc[:, i].to_numpy(dtype=float)
        curves.append(pd.DataFrame({
            "time": grid,
            "survival": surv,
            "profile": label,
        }))
        values.append(simpson_rmst(grid, surv))

    curve_df = pd.concat(curves, ignore_index=True)
    results = pd.DataFrame({
        "profile": profile_labels,
        "tau": float(tau),
        "RMST": values,
    })
    result = {
        "model": model,
        "newdata": newdata.copy(),
        "tau": float(tau),
        "time_grid": grid,
        "curves": curve_df,
        "results": results,
        "treatment": treatment,
    }

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
