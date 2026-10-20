import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from coxrmstsimpson import cox_rmst, plot_cox_rmst

# Make the simulation reproducible
np.random.seed(123)

# Simulate survival data for 200 individuals
n = 200
group = np.random.binomial(1, 0.5, n)

# Group 0 = Control; Group 1 = Treatment
event_time = np.random.exponential(
    scale=1 / (0.10 * np.exp(-0.5 * group))
)
censor_time = np.random.exponential(scale=15, size=n)

# Observed follow-up time and event indicator
df = pd.DataFrame({
    "time": np.minimum(event_time, censor_time),
    "event": (event_time <= censor_time).astype(int),
    "group": group
})

print("\n--- Simulated data ---")
print(df.head())
print("\nNumber of individuals:", len(df))
print("Observed events:", int(df["event"].sum()))

# Fit the Cox proportional hazards model
model = CoxPHFitter()
model.fit(df, duration_col="time", event_col="event")

print("\n--- Cox model results ---")
print(model.summary)

# Predict survival for both groups
newdata = pd.DataFrame({
    "group": [0, 1]
})

print("\n--- Calculating RMST ---")
result = cox_rmst(
    model=model,
    newdata=newdata,
    tau=10,
    n_grid=101
)

print("Result type:", type(result))
print("Result keys:", list(result.keys()))
print("\nRMST results:")
print(result.get("results", "No 'results' entry found."))

# Test the package plotting function
print("\n--- Testing plot_cox_rmst ---")
ax = plot_cox_rmst(result)

# Display the plot in this test script
plt.show()

print("\nSUCCESS: The package test script completed.")
