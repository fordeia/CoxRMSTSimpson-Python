import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from coxrmstsimpson import cox_rmst

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

cph = CoxPHFitter()
cph.fit(df, duration_col="time", event_col="event")

result = cox_rmst(
    cph,
    treatment="group",
    tau=10,
    labels=["Control", "Treatment"],
    title="Restricted Mean Survival Time by Treatment"
)

print(result["results"])