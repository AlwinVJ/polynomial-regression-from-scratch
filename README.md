# Polynomial Regression From Scratch in Python

A practical implementation of **Polynomial Regression from scratch using Python and NumPy**.

The goal of this project is to understand how Polynomial Regression works internally by implementing the algorithm without using machine learning libraries such as Scikit-learn.

---

## Project Overview

Polynomial Regression is a regression technique used to model relationships between variables when the relationship between the input and output is not adequately represented by a straight line.

It extends Linear Regression by transforming the original input features into polynomial features.

For a single input feature \(x\), a polynomial regression model can be represented as:

$$
\hat{y} =
w_0 +
w_1x +
w_2x^2 +
w_3x^3 +
\cdots +
w_nx^n
$$

Where:

* \(x\) = input feature
* \(\hat{y}\) = predicted output
* \(w_0\) = bias/intercept
* \(w_1, w_2, \ldots, w_n\) = model weights
* \(n\) = polynomial degree

Although the model produces a curved relationship with the original input, it is still **linear with respect to its parameters (weights)**.

---

## Objective

The objective of this project is to understand and implement the complete Polynomial Regression workflow from scratch:

1. Load the dataset
2. Prepare the input and target variables
3. Generate polynomial features
4. Initialize model parameters
5. Calculate predictions
6. Calculate the loss
7. Calculate gradients
8. Update weights and bias using Gradient Descent
9. Train the model
10. Generate predictions
11. Visualize the fitted polynomial curve
12. Evaluate the model using \(R^2\)

---

## What This Project Implements

The implementation will contain the following major components:

### 1. Polynomial Feature Transformation

The original input feature:

$$
X
$$

is transformed into:

$$
[X, X^2, X^3, \ldots, X^n]
$$

For example, for degree 3:

$$
X \rightarrow [X, X^2, X^3]
$$

This transformation allows a linear regression algorithm to model nonlinear relationships in the original feature space.

---

### 2. Prediction

After transforming the input features, predictions are calculated using:

$$
\hat{y} = Xw + b
$$

where the transformed \(X\) contains the polynomial features.

---

### 3. Loss Calculation

The model will use Mean Squared Error (MSE) to measure the difference between actual and predicted values.

$$
MSE =
\frac{1}{m}
\sum_{i=1}^{m}
(y_i-\hat{y}_i)^2
$$

---

### 4. Gradient Calculation

The gradients of the loss function with respect to the weights and bias will be calculated using vectorized NumPy operations.

These gradients are then used by Gradient Descent to update the model parameters.

---

### 5. Gradient Descent

The parameters are updated using:

$$
w := w - \alpha \frac{\partial L}{\partial w}
$$

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

where:

* \(w\) = model weights
* \(b\) = bias
* \(\alpha\) = learning rate
* \(L\) = loss function

---

### 6. \(R^2\) Score

The model will also implement the coefficient of determination:

$$
R^2 =
1 -
\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
$$

where:

* \(y_i\) = actual value
* \(\hat{y}_i\) = predicted value
* \(\bar{y}\) = mean of the actual values

The \(R^2\) score measures how much of the variance in the target variable is explained by the model.

---

## Project Structure

```text
Polynomial_Regression_From_Scratch/
│
├── data/
│   └── polynomial_data.csv
│
├── notebooks/
│   └── polynomial_regression.ipynb
│
├── src/
│   ├── __init__.py
│   └── polynomial_regression.py
│
├── tests/
│   └── test_polynomial_regression.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook
* Pytest
* Git
* GitHub

### Machine Learning Concepts

* Polynomial Regression
* Linear Regression
* Polynomial Feature Transformation
* Mean Squared Error
* Gradient Descent
* Partial Derivatives
* Vectorization
* Model Prediction
* \(R^2\) Score

---

## Implementation Philosophy

This project intentionally avoids high-level machine learning libraries such as:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

Instead, the important components are implemented manually using Python and NumPy.

The purpose is not simply to obtain predictions, but to understand what happens internally when a Polynomial Regression model is trained.

---

## Learning Flow

The project follows this pipeline:

```text
Raw Data
   ↓
Input Feature X
   ↓
Polynomial Feature Transformation
   ↓
[X, X², X³, ..., Xⁿ]
   ↓
Initialize Weights & Bias
   ↓
Prediction
   ↓
Calculate Loss
   ↓
Calculate Gradients
   ↓
Gradient Descent
   ↓
Update Parameters
   ↓
Repeat
   ↓
Trained Polynomial Regression Model
   ↓
Predictions
   ↓
R² Evaluation
   ↓
Visualization
```

---

## Example

Suppose the original dataset contains:

```text
X
---
1
2
3
4
```

For a polynomial degree of 3, the feature transformation becomes:

```text
X    X²    X³
1     1     1
2     4     8
3     9    27
4    16    64
```

The model can then learn:

$$
\hat{y} =
w_0 +
w_1X +
w_2X^2 +
w_3X^3
$$

The important idea is that the model is still learning weights using a linear combination of the transformed features.

---

## Future Improvements

Possible extensions include:

* Support for multiple input features
* Different polynomial degrees
* Feature scaling
* Regularization
* Ridge Regression
* Lasso Regression
* Train/test evaluation
* Cross-validation
* Hyperparameter experimentation
* Interactive visualization

---

## Related Project

This project is part of my machine learning algorithms-from-scratch series.

### Linear Regression From Scratch

[Linear Regression From Scratch](https://github.com/AlwinVJ/Linear_Regression_From_Scratch)

### Linear Regression Lab

[Linear Regression Lab](https://alwinvj.github.io/Linear_Regression_Lab/)

---

## Author

**Alwin V J**

Building machine learning algorithms from scratch to understand their mathematical and computational foundations.
