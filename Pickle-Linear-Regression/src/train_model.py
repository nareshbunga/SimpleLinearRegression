"""Train and evaluate the Simple Linear Regression salary model."""

from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Salary_Data.csv"
MODEL_PATH = ROOT / "models" / "linear_regression_model.pkl"
OUTPUT_PATH = ROOT / "outputs" / "metrics.json"
PLOT_PATH = ROOT / "images" / "salary_regression.png"


def main():
    df = pd.read_csv(DATA_PATH)
    required = {"YearsExperience", "Salary"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    df = df.dropna(subset=["YearsExperience", "Salary"])

    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "training_r2": float(model.score(X_train, y_train)),
        "test_r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(df)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model with the scikit-learn version used by this environment.
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    OUTPUT_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    # Plot all observations and the fitted regression line.
    x_line = np.linspace(X["YearsExperience"].min(), X["YearsExperience"].max(), 200)
    y_line = model.predict(pd.DataFrame({"YearsExperience": x_line}))
    plt.figure(figsize=(8, 5))
    plt.scatter(X["YearsExperience"], y, label="Observed salaries")
    plt.plot(x_line, y_line, label="Linear regression line")
    plt.title("Salary vs. Years of Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()

    print("Model saved:", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
