# Salary Prediction using Simple Linear Regression

A small end-to-end machine learning project that trains a **Simple Linear Regression** model and serves salary estimates through a **Streamlit** web application.

## Project overview

The model learns the relationship between:

- **Input feature:** `YearsExperience`
- **Target:** `Salary`

The supplied dataset contains 30 rows. The training script splits the data into 80% training and 20% testing data (`random_state=0`), trains a `LinearRegression` estimator, evaluates it, saves the fitted model, and creates a regression plot.

## Repository structure

```text
Salary-Prediction-Linear-Regression/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── Salary_Data.csv
├── models/
│   └── linear_regression_model.pkl
├── src/
│   └── train_model.py
├── images/
│   └── salary_regression.png
└── outputs/
    └── metrics.json
```

## Setup

Python 3.10 or later is recommended.

Create and activate a virtual environment:

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the model

Run from the repository root:

```bash
python src/train_model.py
```

This creates or refreshes:

- `models/linear_regression_model.pkl`
- `outputs/metrics.json`
- `images/salary_regression.png`

The model file is generated using the installed scikit-learn version. For reliable pickle loading, use the same dependency versions when training and running the app.

## Run the Streamlit application

From the repository root:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser, enter years of experience, and select **Predict salary**.

## Evaluation

The training script reports:

- Training and test R²
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Learned slope and intercept

See `outputs/metrics.json` for the results generated in this project run.

## Model notes

Simple Linear Regression models a linear relationship between one input variable and a continuous target. The fitted model learns its slope and intercept from the training data; the application uses that saved fitted estimator for predictions.

The app warns when an input is outside the experience range in the supplied training dataset. Predictions outside that range are extrapolations and may be unreliable.

## Dataset

`data/Salary_Data.csv` is the dataset supplied for this project. Before making this repository public, confirm that you have permission to redistribute the dataset and that it contains no confidential information.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pickle (model persistence)

## Important security note

Only load pickle files that you created or otherwise trust. Pickle files can execute code during deserialization; do not load untrusted `.pkl` files.
