
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import coxrmstsimpson
from coxrmstsimpson import cox_rmst, plot_cox_rmst

# Confirm that the package is loaded from the installed environment
print("Package location:")
print(coxrmstsimpson.__file__)

# 1. Simulate survival data
np.random.seed(42)
n = 400

gender = np.random.binomial(1, 0.5, n)

# Gender 0 is the reference group
# Gender 1 has a simulated hazard ratio of exp(0.4)
event_time = np.random.exponential(
    scale=1 / (0.10 * np.exp(0.4 * gender)),
    size=n
)

censor_time = np.random.exponential(scale=15, size=n)

df = pd.DataFrame({
    "time": np.minimum(event_time, censor_time),
    "status": (event_time <= censor_time).astype(int),
    "gender": gender
})

print("\nEvents by gender:")
print(df.groupby("gender")["status"].agg(["count", "sum"]))

# 2. Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col="time", event_col="status")

print("\nCox proportional hazards model:")
cph.print_summary()

# 3. Estimate RMST for each gender group using your package
profiles = pd.DataFrame({"gender": [0, 1]})

result = cox_rmst(
    model=cph,
    newdata=profiles,
    tau=10,
    n_grid=101
)

print("\nRestricted mean survival time estimates (tau = 10):")
print(result["results"].to_string(index=False))

# 4. Plot predicted survival curves and shaded RMST areas
ax = plot_cox_rmst(
    result,
    labels=["Gender 0", "Gender 1"],
    show_rmst=True
)

ax.set_title("Predicted Survival by Gender")
plt.tight_layout()
plt.savefig("gender_cox_rmst.png", dpi=150)
plt.show()

print("\nAnalysis completed.")

