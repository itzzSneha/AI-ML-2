# Titanic Data Cleaning and Regression Analysis

This repository contains a Python script for cleaning the Titanic dataset, performing exploratory data analysis, and applying multiple regression modeling to predict passenger fares. The script also includes model evaluation metrics and visualizations for better understanding of the data.

## Features

- **Data Cleaning:**  
  - Handles missing values in `Age` and `Embarked`
  - Drops the `Cabin` column due to excessive missing data
  - Removes duplicate rows
  - Encodes categorical variables (`Sex`, `Embarked`)
  - Standardizes `Age` and `Fare` columns

- **Exploratory Data Analysis:**  
  - Descriptive statistics for numerical and categorical columns
  - Value counts for `Survived`, `Sex`, and `Embarked`
  - Visualizations:
    - Age distribution histogram
    - Survival count bar plot

- **Regression Modeling:**  
  - Multiple linear regression to predict `Fare` using features: `Age`, `Sex`, `Pclass`, `SibSp`, `Parch`, `Embarked`
  - Displays regression coefficients and intercept

- **Model Evaluation:**  
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score

- **Model Interpretation:**  
  - Explains the meaning of regression coefficients

- **Output:**  
  - Saves the cleaned dataset as `Titanic-Dataset-clean.csv`

## Usage

1. Place `Titanic-Dataset.csv` in the same directory as the script.
2. Run the script:
    ```sh
    python Titanic_clean.py
    ```
3. The script will output statistics, model results, and display plots.

## Visualizations

The script generates and displays:
- Age distribution histogram
- Survival count bar plot

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Files

- [`Titanic_clean.py`](c:/Users/sneha/Downloads/Titanic_clean.py): Main script for data cleaning, analysis, and modeling.
- `Titanic-Dataset.csv`: Input dataset (not included).
- `Titanic-Dataset-clean.csv`: Output cleaned dataset.

