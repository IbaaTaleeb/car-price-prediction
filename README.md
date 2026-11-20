# Car Price Prediction with Regression Models

Individual assignment for **Machine Learning and Data Science (ENCS5341)** at Birzeit University.

The goal is to predict car prices using the [Yalla Motors cars dataset](https://www.kaggle.com/datasets/ahmedwaelnasef/cars-dataset/data) (about 6,750 cars scraped from the YallaMotors website), and to compare linear and nonlinear regression models.

## Data cleaning and feature engineering

- **Common currency:** prices were listed in several currencies, so they were converted to USD
- **Numeric columns:** engine capacity, cylinders, horsepower, top speed and seats were cleaned and converted to numbers
- **Missing values:** filled with the most common value for the same car model, then for the same brand; the few rows left (58, under 1%) were dropped
- **Encoding:** label encoding for car name, brand and country
- **Scaling:** Min-Max normalization and standardization of the numerical features
- **Split:** 60% training, 20% validation, 20% test

## Models

**Linear models**
- Linear Regression, **LASSO** (L1) and **Ridge** (L2), with the regularization strength chosen by grid search
- Linear regression **implemented from scratch** in two ways: the closed-form solution (normal equation) and gradient descent. Both gave almost the same results, as expected.

**Nonlinear models**
- **Polynomial regression** with degree 2 to 10
- **Support Vector Regression with an RBF (Gaussian) kernel**

**Feature selection**
- **Forward selection** added one feature at a time based on validation error. It kept six features: horsepower, top speed, cylinders, country, seats and engine capacity.

## Results (validation set)

| Model | R² |
|---|---|
| Linear / LASSO / Ridge | about 0.58 |
| Polynomial, degree 3 | 0.64 |
| Polynomial, degree above 3 | overfitting (negative R²) |
| **RBF kernel (SVR)** | **0.675** |

The **RBF model** was selected as the best and then evaluated once on the test set:

| Metric | Validation | Test |
|---|---|---|
| R² | 0.675 | **0.662** |
| MSE | 0.188 | 0.196 |
| MAE | 0.190 | 0.200 |

The test results are close to the validation results, which shows the model generalizes well to unseen data.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Google Colab

## Files

| File | Description |
|---|---|
| `1203073IbaaTaleeb.ipynb` | Full code: cleaning, models, feature selection, tuning and evaluation |
| `1203073MachineLearningAssignment2Updated.pdf` | Report with the method, charts and results |
