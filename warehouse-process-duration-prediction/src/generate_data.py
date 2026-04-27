import pandas as pd
import numpy as np


def main():
    np.random.seed(42)

    n = 1000

    df = pd.DataFrame({
        "order_lines": np.random.randint(1, 30, n),
        "total_items": np.random.randint(1, 300, n),
        "warehouse_area": np.random.choice(["A", "B", "C", "D"], n),
        "shift": np.random.choice(["early", "late", "night"], n),
        "weekday": np.random.randint(0, 7, n),
        "employee_experience_years": np.round(np.random.uniform(0, 10, n), 1),
        "priority_order": np.random.choice([0, 1], n, p=[0.8, 0.2])
    })

    area_factor = df["warehouse_area"].map({
        "A": 1.0,
        "B": 1.2,
        "C": 1.5,
        "D": 1.8
    })

    shift_factor = df["shift"].map({
        "early": 1.0,
        "late": 1.1,
        "night": 1.25
    })

    df["process_duration_min"] = (
        5
        + df["order_lines"] * 1.3
        + df["total_items"] * 0.08
        + area_factor * 4
        + shift_factor * 3
        - df["employee_experience_years"] * 0.7
        + df["priority_order"] * 5
        + np.random.normal(0, 5, n)
    ).round(2)

    df.to_csv("data/warehouse_process_data.csv", index=False)

    print("Data generated successfully.")


if __name__ == "__main__":
    main()