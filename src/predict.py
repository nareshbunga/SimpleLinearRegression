"""Load the trained model and predict salary for experience in years."""
from pathlib import Path
import argparse
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.pkl"


def predict_salary(years_experience: float) -> float:
    if years_experience < 0:
        raise ValueError("Years of experience cannot be negative.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    prediction = model.predict(np.array([[years_experience]], dtype=float))
    return float(prediction[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experience", type=float, required=True,
        help="Years of work experience (for example: 5 or 12.5)."
    )
    args = parser.parse_args()
    salary = predict_salary(args.experience)
    print(f"Predicted salary for {args.experience:g} years of experience: {salary:,.2f}")


if __name__ == "__main__":
    main()
