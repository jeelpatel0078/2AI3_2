# Insurance Charges Prediction

## Project Title

Insurance Charges Prediction using Linear Regression

## Team Member Details

- Team Member 1: [Machhi Ruchit] - [Data Loading and Preprocessing]
- Team Member 2: [Gandhi Laksh] - [Model Building]
- Team Member 3: [Jaiswal Pushkar] - [Model Evaluation]
- Team Member 4: [Kammila Charan Paul] - [README file]
- Team Member 5: [Kanakshri Joshi] - [README file]
- Team Member 6: [Kavya Sankhala] - [Data Splitting]
- Team Member 7: [Khandivar Jeel] - [Prediction making]

> Replace the placeholders above with the actual project team members.

## Problem Statement

This project predicts individual health insurance charges using demographic and lifestyle data. The goal is to estimate the insurance cost for a person based on age, sex, BMI, number of children, smoking status, and residential region.

## Dataset Description

The dataset is stored in `insurance_data_linear.csv` and contains the following columns:

- `age`: Age of the policyholder
- `sex`: Gender of the policyholder (`male`, `female`)
- `bmi`: Body Mass Index value
- `children`: Number of children covered by the insurance policy
- `smoker`: Smoking status (`yes`, `no`)
- `region`: Residential area (`southwest`, `southeast`, `northwest`, `northeast`)
- `charges`: Insurance charges (target variable)

The dataset contains 1,338 records and no missing values.

## Data Preprocessing Steps

1. Load the CSV file using `pandas`.
2. Copy the dataset for preprocessing.
3. Encode categorical variables (`sex`, `smoker`, `region`) using `LabelEncoder`.
4. Separate features (`X`) and target (`y`), where `charges` is the target.
5. Split the data into training and testing sets with a 70/30 ratio using `train_test_split`.
6. Scale feature values using `StandardScaler`.

## Model Used and Training Details

- Model: `LinearRegression` from `scikit-learn`
- Training data size: 936 samples
- Testing data size: 402 samples
- Feature scaling: Standard scaling applied to all input features
- Training process:
  - Fit the linear regression model on scaled training features
  - Predict on both training and testing sets

## Model Evaluation Results

### Training Metrics

- Mean Absolute Error (MAE): $4,251.53
- Root Mean Squared Error (RMSE): $6,144.20
- R² Score: 0.7423

### Testing Metrics

- Mean Absolute Error (MAE): $4,155.24
- Root Mean Squared Error (RMSE): $5,814.25
- R² Score: 0.7694

### Model Coefficients

- `age`: 3693.2242
- `sex`: 54.8056
- `bmi`: 2064.8559
- `children`: 514.3279
- `smoker`: 9592.7960
- `region`: -363.3593

### Example Prediction

For a new customer with:

- age: 25
- sex: male
- bmi: 28.5
- children: 1
- smoker: no
- region: northeast

Predicted insurance charges: **$4,529.80**

## GitHub Collaboration Summary

- This repository contains the project script `main.py` and the dataset file `insurance_data_linear.csv`.
- Collaborators can contribute by updating the README, improving preprocessing, testing alternative models, and adding data visualization.
- Use GitHub branches and pull requests to manage feature work.
- Include clear commit messages and code comments for reproducibility.

## Conclusion

The linear regression model provides a good starting point for predicting insurance charges from demographic and health-related features. The evaluation results show reasonable predictive accuracy, especially on the testing set. Future improvements can include more advanced regression techniques, feature engineering, and hyperparameter tuning to reduce prediction error.
