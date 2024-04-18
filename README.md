# Diabetes Readmission Prediction

This repository contains code for analyzing and predicting diabetes readmission using machine learning techniques. 

## Overview

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria:

1. It is an inpatient encounter (a hospital admission).
2. It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis.
3. The length of stay was at least 1 day and at most 14 days.
4. Laboratory tests were performed during the encounter.
5. Medications were administered during the encounter.

The data contains attributes such as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab tests performed, HbA1c test result, diagnosis, number of medications, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before hospitalization, etc.

## Requirements

The following Python libraries are required to run the code:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- catboost
- xgboost

These can be installed via pip:


## Usage

1. Clone the repository to your local machine:


2. Navigate to the project directory:


3. Run the main script `svm.py`:
   
4. Run the main script `random_forest.py`:

5. 
This script performs data preprocessing, feature engineering, model training, evaluation, and interpretation.

## Data Visualization

The repository includes various data visualization techniques, such as histogram plots, bar plots, and pie charts, to understand the distribution of data and explore relationships between variables.

## Model Training and Evaluation

The repository trains multiple machine learning models, including Support Vector Machine (SVM), Random Forest, and XGBoost, to predict diabetes readmission. The models are evaluated using cross-validation and appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.

## Model Interpretation

Feature importance analysis is performed to identify relevant features influencing readmission rates. This helps gain insights into factors affecting readmission and aids in clinical decision-making.

## Contributors

Feel free to contribute to this project by opening issues or pull requests.



