# Demand Forecasting with scikit-learn

## Goal
This project predicts product demand based on historical sales and inventory-related features.

## Business Context
Demand forecasting can help companies improve stock planning, reduce shortages and avoid overstock.

## Dataset
The current version uses synthetic data with features such as:
- day of week
- month
- stock level
- sales last week
- promotion flag

## Model
I used a RandomForestRegressor from scikit-learn.

## Evaluation
The model is evaluated using:
- Mean Absolute Error
- R² Score

## How to run

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
python src/predict.py