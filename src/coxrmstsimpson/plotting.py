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
    """Plot survival curves with user-supplied category labels."""
    if not isinstance(result, dict) or "curves" not in result or "tau" not in result:
        raise TypeError("result must be returned by cox_rmst().")

    curves = result["curves"]
    profiles = list(curves.groupby("profile", sort=False))
    results_table = result.get("results")

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(profiles):
            raise ValueError(
                f"Provide one label per profile ({len(profiles)} required)."
            )

    if show_rmst and (
        results_table is None
        or not {"profile", "RMST"}.issubset(results_table.columns)
    ):
        raise ValueError("RMST results are unavailable.")

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
                raise ValueError(f"Missing RMST estimate for {profile!r}.")
            curve_label += f" (RMST = {matching.iloc[0]:.3f})"

        ax.plot(data["time"], data["survival"], label=curve_label)
        ax.fill_between(data["time"], data["survival"], 0, alpha=0.18)

    ax.set(
        xlim=(0, result["tau"]),
        ylim=(0, 1.02),
        xlabel="Time",
        ylabel="Predicted survival probability",
        title=title or f"Restricted mean survival time (tau = {result['tau']:g})",
    )
    ax.legend()
    ax.grid(alpha=0.25)

    if show:
        plt.show()

    return ax
