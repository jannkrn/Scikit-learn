import pandas as pd
import joblib


def main():
    # Load trained model
    model = joblib.load("models/demand_model.pkl")

    # New example data for prediction
    new_data = pd.DataFrame({
        "day_of_week": [1],          # 0 = Monday, 6 = Sunday
        "month": [12],
        "stock": [500],
        "sales_last_week": [120],
        "promotion": [1]             # 1 = promotion active, 0 = no promotion
    })

    # Make prediction
    prediction = model.predict(new_data)

    print(f"Predicted demand: {prediction[0]:.2f} units")


if __name__ == "__main__":
    main()