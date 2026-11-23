"""Train a simple linear regression model using Salary_Data.csv."""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Salary_Data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.pkl"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    required_columns = {"YearsExperience", "Salary"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}")

    X = dataset[["YearsExperience"]]
    y = dataset["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "train_r2": float(model.score(X_train, y_train)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )

    predictions = pd.DataFrame({
        "YearsExperience": X_test["YearsExperience"].to_numpy(),
        "ActualSalary": y_test.to_numpy(),
        "PredictedSalary": y_pred,
    })
    predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    print("Model training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Slope (m): {model.coef_[0]:,.2f}")
    print(f"Intercept (c): {model.intercept_:,.2f}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"MAE: {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")


if __name__ == "__main__":
    main()
