# Medical Insurance Statistical Analysis Dashboard

## Project Overview

This project presents an interactive statistical analysis dashboard built using Python and Streamlit. The dashboard analyzes a medical insurance dataset and provides exploratory data analysis, hypothesis testing, and multiple linear regression modeling.

The application is organized into three main sections:

- Exploratory Data Analysis (EDA)
- Hypothesis Testing
- Multiple Linear Regression

Users can interact with filters, statistical tests, visualizations, regression diagnostics, and a live medical-charge prediction tool.

---

## Dataset

The project uses the Medical Insurance dataset.

The dataset contains information about individuals and their medical insurance charges.

### Variables

| Variable | Description |
|---|---|
| age | Age of the individual |
| sex | Gender of the individual |
| bmi | Body Mass Index |
| children | Number of children/dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges |

The dataset is fetched directly from the following public GitHub URL:

https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

---

## Features

### 1. Exploratory Data Analysis

The dashboard provides:

- Dataset overview
- Interactive data filtering
- Descriptive statistics
- Mean
- Median
- Standard deviation
- Interquartile range (IQR)
- Skewness
- Kurtosis
- Distribution plots
- Box plots
- Bivariate analysis
- Correlation matrix

### 2. Hypothesis Testing

The dashboard performs:

- Shapiro-Wilk normality tests
- Levene's test for equality of variances
- Independent two-group comparison
- Independent samples t-test when appropriate
- Mann-Whitney U test when normality assumptions are not satisfied
- One-way ANOVA
- Group-level descriptive statistics
- Box plots for group comparisons

### 3. Multiple Linear Regression

A multiple linear regression model is fitted with:

**Response variable:**
- charges

**Predictor variables:**
- age
- bmi
- children
- sex
- smoker
- region

The regression section includes:

- OLS regression model
- Regression coefficients
- Standard errors
- t-statistics
- p-values
- 95% confidence intervals
- R-squared
- Adjusted R-squared
- F-statistic and model p-value
- Residual-vs-fitted diagnostic
- Q-Q plot
- Jarque-Bera residual normality test
- Variance Inflation Factor (VIF)
- Complete Statsmodels OLS summary
- Live medical-charge prediction
- 95% mean confidence interval
- 95% prediction interval

---

## Statistical Findings

### Exploratory Data Analysis

The dataset contains both numerical and categorical variables. The dashboard shows the distributions of age, BMI, children, and medical charges and provides summary statistics including measures of central tendency, variability, skewness, and kurtosis.

Medical charges show substantial variability and a right-skewed distribution, indicating that a relatively small number of individuals have considerably higher insurance charges.

The correlation analysis helps identify relationships between numerical variables and medical charges.

### Hypothesis Testing

The dashboard first checks the assumptions required for the two-group comparison using Shapiro-Wilk and Levene's tests.

When the normality assumption is not satisfied, the Mann-Whitney U test is used instead of the independent samples t-test.

The dashboard also performs a one-way ANOVA to determine whether mean medical charges differ across the selected groups.

Statistical decisions are made using a significance level of:

**α = 0.05**

### Multiple Linear Regression

A multiple linear regression model is used to explain medical insurance charges using age, BMI, number of children, sex, smoking status, and region.

The regression output provides coefficient estimates and significance tests for each predictor, along with overall model performance measures such as R-squared, adjusted R-squared, and the F-test.

The diagnostic analysis indicates that the residuals do not follow a normal distribution. The Jarque-Bera test gives a very small p-value, providing evidence against the null hypothesis of normally distributed residuals.

The VIF diagnostic indicates that all predictors have VIF values below 5, suggesting that there is no serious multicollinearity among the predictors.

The residual diagnostics also indicate that some linear regression assumptions may not be fully satisfied, particularly regarding residual normality and constant variance.

---

## Installation

Clone or download this repository and navigate to the project directory.

Install the required Python packages:

```bash
pip install -r requirements.txt