import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    df = pd.read_csv("data/warehouse_process_data.csv")

    X = df.drop("process_duration_min", axis=1)
    y = df["process_duration_min"]

    categorical_features = ["warehouse_area", "shift"]

    numeric_features = [
        "order_lines",
        "total_items",
        "weekday",
        "employee_experience_years",
        "priority_order"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print(f"R² Score: {r2:.2f}")

    joblib.dump(pipeline, "models/process_duration_model.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    main()