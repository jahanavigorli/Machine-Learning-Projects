# Wine Quality Dataset – Normalization Techniques

## 1. Project Overview

This project demonstrates the application and comparison of five different normalization and scaling techniques on the Red Wine Quality dataset using Python.

The main objective is to understand how different scaling techniques transform numerical features and how normalization affects data distribution and machine learning model performance.

## 2. Dataset Description

**Dataset:** Red Wine Quality Dataset

**Source:** UCI Machine Learning Repository

The dataset contains physicochemical properties of red wine samples and their corresponding quality scores.

- Total Features: 11 input features
- Target Variable: `quality`
- File Format: CSV
- Separator: Semicolon (;)

### Input Features

1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

**Target Variable:** Quality

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 4. Normalization Techniques

Five scaling techniques are implemented in this experiment.

### 1. Min-Max Scaling

Transforms feature values into a fixed range, generally between 0 and 1.

Formula:

X' = (X - Xmin) / (Xmax - Xmin)

### 2. Standard Scaling

Transforms data to have a mean of 0 and a standard deviation of 1.

Formula:

X' = (X - Mean) / Standard Deviation

### 3. Robust Scaling

Uses the median and interquartile range (IQR) to scale the data. It is less sensitive to outliers.

Formula:

X' = (X - Median) / IQR

### 4. MaxAbs Scaling

Divides each feature by its maximum absolute value
