# Simple Linear Regression - Salary Prediction

A beginner-friendly Simple Linear Regression project that predicts salary from
years of experience using the supplied `Salary_Data.csv` dataset.

## 1. Folder structure

```text
SimpleLinearRegression_Project/
├── data/
│   └── Salary_Data.csv
├── models/
│   └── linear_regression_model.pkl
├── outputs/
│   ├── plots/
│   │   ├── training_set.png
│   │   └── test_set.png
│   └── reports/
│       ├── metrics.json
│       └── test_predictions.csv
├── src/
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 2. Requirements

- Python 3.10 or newer (Python 3.11 or 3.12 is recommended for broad ML-library compatibility).
- pip

## 3. Setup

Open a terminal in the project root folder.

```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Train the model

Run from the project root:

```bash
python src/train.py
```

This trains a LinearRegression model using an 80/20 train-test split
(`random_state=0`) and saves the model and evaluation reports.

## 5. Predict a salary

Example: predict the salary for 12 years of experience.

```bash
python src/predict.py --experience 12
```

Try another value:

```bash
python src/predict.py --experience 20
```

The model uses the relationship:

`Salary = slope * YearsExperience + intercept`

The slope and intercept are learned from the supplied dataset during training.
The model should be used cautiously for values far outside the training data
(extrapolation).

## 6. Generate visualizations

```bash
python src/visualize.py
```

The training-set and test-set plots are saved under `outputs/plots/`.

## 7. Evaluation

The training script reports:
- **R² score:** goodness of fit; it is not itself a direct measurement of bias or variance.
- **MAE:** average absolute prediction error.
- **MSE:** average squared prediction error.
- **RMSE:** square root of MSE, in the target's units.

The generated values are saved to `outputs/reports/metrics.json`.
The held-out test predictions are saved to `outputs/reports/test_predictions.csv`.

## 8. Important notes

- All file paths are built relative to the project folder; no machine-specific
  `E:\...` path is required.
- The `.pkl` model is a Python pickle. Load pickle files only from sources you trust.
- Pickled scikit-learn models can be version-sensitive. If loading the bundled
  model raises a compatibility warning/error, install compatible dependencies
  and retrain with `python src/train.py`.
- The original uploaded model was re-trained from the uploaded CSV and saved
  with the current environment's scikit-learn version.
