import pandas as pd
import joblib


def main():
    model = joblib.load("models/process_duration_model.pkl")

    new_order = pd.DataFrame({
        "order_lines": [12],
        "total_items": [120],
        "warehouse_area": ["C"],
        "shift": ["late"],
        "weekday": [2],
        "employee_experience_years": [3.5],
        "priority_order": [1]
    })

    prediction = model.predict(new_order)

    print(f"Predicted process duration: {prediction[0]:.2f} minutes")


if __name__ == "__main__":
    main()