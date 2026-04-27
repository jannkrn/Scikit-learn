# Warehouse Process Duration Prediction

## Project Goal

This project predicts the duration of warehouse processes based on order and operational features using scikit-learn.

## Business Context

In warehouse operations, estimating process duration can support workforce planning, bottleneck detection and operational efficiency.

For example, if a company can estimate how long certain order types take, it can plan employees, shifts and priorities more effectively.

## Dataset

The dataset is synthetically generated and contains the following features:

- order_lines
- total_items
- warehouse_area
- shift
- weekday
- employee_experience_years
- priority_order

The target variable is:

- process_duration_min

## Machine Learning Approach

This project uses a scikit-learn Pipeline with:

- OneHotEncoder for categorical features
- ColumnTransformer for preprocessing
- RandomForestRegressor for prediction

## Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error
- R² Score

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
python src/predict.py