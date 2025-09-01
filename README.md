# Telco Customer Churn Analysis

## Overview
This project analyzes customer churn data from a telecommunications company using machine learning techniques. The goal is to identify patterns and factors that contribute to customer churn and build a predictive model using logistic regression.

## Features
- **Exploratory Data Analysis (EDA)**: Visualizations of churn patterns and relationships with key variables
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling
- **Machine Learning**: Logistic regression model for churn prediction
- **Model Evaluation**: Confusion matrix and classification report
- **Feature Importance Analysis**: Identification of the most influential factors driving churn

## Dataset
The analysis uses the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset containing information about telecom customers, including:
- Demographic information
- Account details (tenure, contract type)
- Service usage patterns
- Billing information
- Churn status (target variable)

## Key Visualizations
1. **Churn Distribution**: Overall distribution of churned vs. retained customers
2. **Tenure vs Churn**: Relationship between customer tenure and churn likelihood
3. **Monthly Charges vs Churn**: Analysis of how monthly charges affect churn
4. **Contract Type Analysis**: Churn patterns across different contract types
5. **Correlation Heatmap**: Relationships between all variables after encoding

## Dependencies
- pandas
- matplotlib
- seaborn
- scikit-learn
- numpy

## Installation
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

## Usage
1. Place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in your working directory
2. Run the script to:
   - Load and preprocess the data
   - Generate exploratory visualizations
   - Train a logistic regression model
   - Evaluate model performance
   - Identify the most influential features affecting churn

## Model Performance
The logistic regression model provides:
- Confusion matrix showing true/false positives and negatives
- Classification report with precision, recall, and F1 scores
- Feature importance ranking showing which factors most significantly impact churn predictions

## Key Insights
The analysis reveals the most influential features affecting customer churn, helping identify areas where the company could focus retention efforts.

## File Structure
```
├── churn_analysis.py          # Main analysis script
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── README.md                  # This file
```

 ## Results Interpretation
- Positive coefficients indicate features that increase churn likelihood
- Negative coefficients indicate features that decrease churn likelihood
- The absolute value of coefficients indicates the strength of influence

This analysis provides actionable insights for developing targeted customer retention strategies in the telecommunications industry.
