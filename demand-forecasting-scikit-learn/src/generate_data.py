import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "day_of_week": np.random.randint(0, 7, n),
    "month": np.random.randint(1, 13, n),
    "stock": np.random.randint(50, 1000, n),
    "sales_last_week": np.random.randint(10, 300, n),
    "promotion": np.random.randint(0, 2, n),
})

df["demand"] = (
    df["sales_last_week"] * 0.6
    + df["promotion"] * 40
    + df["month"].isin([11, 12]).astype(int) * 30
    + np.random.normal(0, 20, n)
).round()

df.to_csv("Data/demand_data.csv", index=False)