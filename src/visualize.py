"""Create training-set and test-set regression plots."""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Salary_Data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.pkl"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"


def main():
    dataset = pd.read_csv(DATA_PATH)
    X = dataset[["YearsExperience"]]
    y = dataset["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    x_line = np.linspace(X.min().iloc[0], X.max().iloc[0], 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for name, x_values, y_values, title in [
        ("training_set.png", X_train, y_train, "Training Set"),
        ("test_set.png", X_test, y_test, "Test Set"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.scatter(x_values.iloc[:, 0], y_values, label="Observed data")
        plt.plot(x_line, y_line, label="Regression line")
        plt.title(f"Salary vs Experience - {title}")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / name, dpi=150)
        plt.close()
        print(f"Saved: {PLOT_DIR / name}")


if __name__ == "__main__":
    main()
