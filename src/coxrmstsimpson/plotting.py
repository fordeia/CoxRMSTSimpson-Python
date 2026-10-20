"""Plot predicted survival curves and shaded RMST areas."""
import matplotlib.pyplot as plt


def plot_cox_rmst(
    result,
    ax=None,
    labels=None,
    show_rmst=False,
    title=None,
    show=True,
):
    """Plot predicted survival curves and shade their RMST areas.

    Parameters
    ----------
    result : dict
        Result returned by cox_rmst().
    ax : matplotlib.axes.Axes, optional
        Existing axes on which to draw the plot.
    labels : sequence of str, optional
        Custom labels for profiles in the order plotted.
    show_rmst : bool, default False
        Include each profile's RMST estimate in the legend.
    title : str, optional
        Custom title. Defaults to a title containing tau.
    show : bool, default True
        Call matplotlib.pyplot.show() before returning.
    """
    if not isinstance(result, dict) or "curves" not in result or "tau" not in result:
        raise TypeError("result must be returned by cox_rmst().")

    curves = result["curves"]
    profiles = list(curves.groupby("profile", sort=False))

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(profiles):
            raise ValueError(
                f"Provide one custom label for each profile ({len(profiles)} required)."
            )

    results_table = result.get("results")
    if show_rmst:
        if results_table is None or not {"profile", "RMST"}.issubset(results_table.columns):
            raise ValueError("RMST results are unavailable in result['results'].")

    if ax is None:
        _, ax = plt.subplots()

    for i, (profile, data) in enumerate(profiles):
        data = data.sort_values("time")
        curve_label = str(labels[i]) if labels is not None else str(profile)

        if show_rmst:
            matching = results_table.loc[
                results_table["profile"] == profile, "RMST"
            ]
            if matching.empty:
                raise ValueError(f"Could not find an RMST estimate for profile {profile!r}.")
            curve_label = f"{curve_label} (RMST = {matching.iloc[0]:.3f})"

        ax.plot(data["time"], data["survival"], label=curve_label)
        ax.fill_between(data["time"], data["survival"], 0, alpha=0.18)

    default_title = f"Restricted mean survival time (tau = {result['tau']:g})"
    ax.set(
        xlim=(0, result["tau"]),
        ylim=(0, 1.02),
        xlabel="Time",
        ylabel="Predicted survival probability",
        title=title if title is not None else default_title,
    )
    ax.legend()
    ax.grid(alpha=0.25)

    if show:
        plt.show()

    return ax
