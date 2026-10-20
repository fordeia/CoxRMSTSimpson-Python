
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from coxrmstsimpson import cox_rmst, plot_cox_rmst

# Reproducible simulated survival data
np.random.seed(42)
n = 200

gender = np.random.binomial(1, 0.5, n)
event_time = np.random.exponential(
    scale=1 / (0.10 * np.exp(0.4 * gender))
)
censor_time = np.random.exponential(scale=15, size=n)

df = pd.DataFrame({
    "time": np.minimum(event_time, censor_time),
    "event": (event_time <= censor_time).astype(int),
    "gender": gender,
})

# Fit the Cox proportional hazards model
cox_model = CoxPHFitter()
cox_model.fit(df, duration_col="time", event_col="event")

# Predict for the two groups
newdata = pd.DataFrame({"gender": [0, 1]})

result = cox_rmst(
    model=cox_model,
    newdata=newdata,
    tau=10,
    n_grid=101,
)

# Plot using custom labels and RMST values in the legend
ax = plot_cox_rmst(
    result,
    labels=["Female", "Male"],
    show_rmst=True,
)

legend_labels = [item.get_text() for item in ax.get_legend().get_texts()]
print("Legend labels:", legend_labels)

assert len(legend_labels) == 2
assert legend_labels[0].startswith("Female (RMST = ")
assert legend_labels[1].startswith("Male (RMST = ")

print("PASS: Actual Cox model, custom labels, and RMST legend work.")
