
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import coxrmstsimpson
from coxrmstsimpson import cox_rmst

print("Package location:", coxrmstsimpson.__file__)

np.random.seed(42)
n = 400
gender = np.random.binomial(1, 0.5, n)

event_time = np.random.exponential(
    scale=1 / (0.10 * np.exp(0.4 * gender)), size=n
)
censor_time = np.random.exponential(scale=15, size=n)

df = pd.DataFrame({
    "time": np.minimum(event_time, censor_time),
    "status": (event_time <= censor_time).astype(int),
    "gender": gender,
})

cph = CoxPHFitter()
cph.fit(df, duration_col="time", event_col="status")

print("\nCox proportional hazards model:")
cph.print_summary()

# One package call: calculate RMST, print results, and create one labeled plot
result = cox_rmst(
    model=cph,
    treatment="gender",
    tau=10,
    n_grid=101,
    labels=["Female", "Male"],
    title="Predicted Survival by Gender",
    show_rmst=True,
    show_plot=True,
)

result["figure"].tight_layout()
result["figure"].savefig("gender_cox_rmst.png", dpi=150)

print("\nAnalysis completed.")
