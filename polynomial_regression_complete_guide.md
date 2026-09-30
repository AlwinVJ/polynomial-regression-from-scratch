# Polynomial Regression — Theory, Mathematics, and From-Scratch Implementation

> A practical and mathematical guide to understanding Polynomial Regression before implementing it from scratch with Python and NumPy.

> **Equation notation:** Mathematical expressions are formatted as display equations using LaTeX for clearer rendering in GitHub and other Markdown viewers that support math.

---

## Table of Contents

1. [What Is Regression?](#1-what-is-regression)
2. [Why Linear Regression Is Not Always Enough](#2-why-linear-regression-is-not-always-enough)
3. [What Is Polynomial Regression?](#3-what-is-polynomial-regression)
4. [The Polynomial Regression Equation](#4-the-polynomial-regression-equation)
5. [Why Polynomial Regression Is Still Linear Regression](#5-why-polynomial-regression-is-still-linear-regression)
6. [Polynomial Feature Transformation](#6-polynomial-feature-transformation)
7. [A Numerical Example](#7-a-numerical-example)
8. [From Features to the Design Matrix](#8-from-features-to-the-design-matrix)
9. [Prediction](#9-prediction)
10. [Loss Function](#10-loss-function)
11. [Why We Need Gradients](#11-why-we-need-gradients)
12. [Deriving the Gradient](#12-deriving-the-gradient)
13. [Gradient Descent](#13-gradient-descent)
14. [The Complete Training Process](#14-the-complete-training-process)
15. [Closed-Form Solution](#15-closed-form-solution)
16. [Why Use Gradient Descent in a From-Scratch Project?](#16-why-use-gradient-descent-in-a-from-scratch-project)
17. [Multiple Input Features](#17-multiple-input-features)
18. [Polynomial Degree](#18-polynomial-degree)
19. [Underfitting and Overfitting](#19-underfitting-and-overfitting)
20. [Bias and Variance](#20-bias-and-variance)
21. [Feature Scaling](#21-feature-scaling)
22. [Regularization](#22-regularization)
23. [Ridge and Lasso with Polynomial Features](#23-ridge-and-lasso-with-polynomial-features)
24. [Model Evaluation](#24-model-evaluation)
25. [R² Score](#25-r2-score)
26. [Adjusted R²](#26-adjusted-r2)
27. [MAE, MSE, and RMSE](#27-mae-mse-and-rmse)
28. [Training, Validation, and Test Sets](#28-training-validation-and-test-sets)
29. [Polynomial Regression vs Linear Regression](#29-polynomial-regression-vs-linear-regression)
30. [Polynomial Regression vs Other Nonlinear Models](#30-polynomial-regression-vs-other-nonlinear-models)
31. [Important Assumptions](#31-important-assumptions)
32. [Common Mistakes](#32-common-mistakes)
33. [From Mathematics to NumPy](#33-from-mathematics-to-numpy)
34. [Recommended Class Design](#34-recommended-class-design)
35. [Complete Conceptual Pipeline](#35-complete-conceptual-pipeline)
36. [Practical Experiment Plan](#36-practical-experiment-plan)
37. [Key Takeaways](#37-key-takeaways)
38. [Further Topics](#38-further-topics)

---

# 1. What Is Regression?

Regression is a supervised machine learning problem where the goal is to predict a **continuous numerical value**.

Examples:

- Predicting salary from years of experience
- Predicting house price from area
- Predicting temperature from time
- Predicting sales from advertising expenditure
- Predicting fuel consumption from engine characteristics

Suppose we have:

$$
X = \text{input feature}
$$

and

$$
Y = \text{target}
$$

A regression model attempts to learn a function:

$$
f(X) \approx Y
$$

After training, the model can produce a prediction:

$$
\hat{Y} = f(X)
$$

where \(\hat{Y}\) means the model's predicted value.

---

# 2. Why Linear Regression Is Not Always Enough

Simple Linear Regression assumes that the relationship between \(X\) and \(Y\) can be approximated by a straight line:

$$
\hat{y} = w_0 + w_1x
$$

For example:

```text
y
│
│             •
│          •
│       •
│    •
│ •
└────────────────── x
```

This works well when the underlying relationship is approximately linear.

But many real relationships are curved.

For example:

$$
y = 2x^2 + 3x + 5
$$

A straight line cannot represent this relationship accurately over a sufficiently large range of \(x\).

We therefore need a model that can represent curvature.

This leads to Polynomial Regression.

---

# 3. What Is Polynomial Regression?

Polynomial Regression is a regression technique that models the relationship between an input variable and a continuous target using polynomial terms of the input.

Instead of using only:

$$
x
$$

we can include:

$$
x^2,\ x^3,\ x^4,\ldots,x^n
$$

The resulting model can represent curves rather than only straight lines.

For example, a degree-2 polynomial is:

$$
\hat{y} = w_0 + w_1x + w_2x^2
$$

A degree-3 polynomial is:

$$
\hat{y} = w_0 + w_1x + w_2x^2 + w_3x^3
$$

A degree-4 polynomial is:

$$
\hat{y} = w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4
$$

The highest exponent determines the **polynomial degree**.

---

# 4. The Polynomial Regression Equation

For a polynomial of degree \(n\):

$$
\boxed{
\hat{y} = w_0 + w_1x + w_2x^2 + \cdots + w_nx^n
}
$$

An equivalent compact form is:

$$
\boxed{
\hat{y} = w_0 + \sum_{j=1}^{n} w_jx^j
}
$$

Where:

| Symbol | Meaning |
|---|---|
| \(x\) | Input feature |
| \(\hat{y}\) | Predicted output |
| \(w_0\) | Bias/intercept |
| \(w_j\) | Weight associated with the \(j\)-th polynomial feature |
| \(n\) | Polynomial degree |

### Example: Degree 3

For \(n=3\):

$$
\hat{y}=w_0+w_1x+w_2x^2+w_3x^3
$$

The model has four parameters:

$$
w_0,\ w_1,\ w_2,\ w_3
$$

---

# 5. Why Polynomial Regression Is Still Linear Regression

This is one of the most important ideas in Polynomial Regression.

At first glance:

$$
\hat{y}=w_0+w_1x+w_2x^2+w_3x^3
$$

looks like a nonlinear model because it contains \(x^2\) and \(x^3\).

However, the model is **linear in its parameters**.

The parameters are:

$$
w_0,w_1,w_2,w_3
$$

and each appears only to the first power.

We can define transformed features:

$$
x_1=x
$$

$$
x_2=x^2
$$

$$
x_3=x^3
$$

Then:

$$
\hat{y}=w_0+w_1x_1+w_2x_2+w_3x_3
$$

This has exactly the same form as Multiple Linear Regression.

So Polynomial Regression can be understood as:

```text
Original feature
       ↓
Polynomial feature transformation
       ↓
Multiple linear regression
```

This distinction is extremely important:

> Polynomial Regression is nonlinear with respect to the original input \(x\), but linear with respect to the model parameters \(w\).

---

# 6. Polynomial Feature Transformation

The key operation is transforming the original feature into polynomial features.

Suppose:

$$
X =
\begin{bmatrix}
1\\
2\\
3\\
4
\end{bmatrix}
$$

For degree 3, transform it into:

$$
X_{poly}=
\begin{bmatrix}
1 & 1^2 & 1^3\\
2 & 2^2 & 2^3\\
3 & 3^2 & 3^3\\
4 & 4^2 & 4^3
\end{bmatrix}
$$

Therefore:

$$
X_{poly}=
\begin{bmatrix}
1 & 1 & 1\\
2 & 4 & 8\\
3 & 9 & 27\\
4 & 16 & 64
\end{bmatrix}
$$

The transformation is:

$$
X \rightarrow [X,X^2,X^3]
$$

For degree \(n\):

$$
X \rightarrow [X,X^2,\ldots,X^n]
$$

---

# 7. A Numerical Example

Suppose our data is:

| \(x\) | \(y\) |
|---:|---:|
| 1 | 10 |
| 2 | 17 |
| 3 | 28 |
| 4 | 43 |
| 5 | 62 |

Imagine that the underlying relationship is approximately:

$$
y=2x^2+3x+5
$$

Let's verify:

For \(x=1\):

$$
y=2(1)^2+3(1)+5=10
$$

For \(x=2\):

$$
y=2(2)^2+3(2)+5=19
$$

If the actual observations contain noise, they will not exactly lie on the theoretical curve.

The important point is that a degree-2 polynomial gives us:

$$
\hat{y}=w_0+w_1x+w_2x^2
$$

The model learns \(w_0,w_1,w_2\) from the data.

---

# 8. From Features to the Design Matrix

For degree 3, include an intercept column.

The transformed matrix becomes:

$$
X_{poly}=
\begin{bmatrix}
1 & x_1 & x_1^2 & x_1^3\\
1 & x_2 & x_2^2 & x_2^3\\
\vdots & \vdots & \vdots & \vdots\\
1 & x_m & x_m^2 & x_m^3
\end{bmatrix}
$$

The first column represents the intercept.

We can write:

$$
\theta=
\begin{bmatrix}
w_0\\
w_1\\
w_2\\
w_3
\end{bmatrix}
$$

Then the entire prediction operation becomes:

$$
\boxed{
\hat{Y}=X_{poly}\theta
}
$$

If the bias is stored separately, the implementation can instead use:

$$
\boxed{
\hat{Y}=X_{poly}W+b
}
$$

Both formulations represent the same basic model.

---

# 9. Prediction

Suppose we have a degree-2 model:

$$
\hat{y}=w_0+w_1x+w_2x^2
$$

Given:

$$
w_0=5,\quad w_1=3,\quad w_2=2
$$

and:

$$
x=4
$$

Then:

$$
\hat{y}=5+3(4)+2(4^2)
$$

$$
=5+12+32
$$

$$
=49
$$

Therefore:

$$
\boxed{\hat{y}=49}
$$

For multiple observations, NumPy can perform the same calculation using vectorized matrix operations.

---

# 10. Loss Function

The model needs a way to measure how wrong its predictions are.

For regression, Mean Squared Error (MSE) is commonly used:

$$
\boxed{
MSE=
\frac{1}{m}
\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
}
$$

where:

- \(m\) = number of observations
- \(y_i\) = actual value
- \(\hat{y}_i\) = predicted value

### Why square the errors?

Suppose an observation has:

$$
y=10,\quad \hat{y}=8
$$

The error is:

$$
10-8=2
$$

Another observation:

$$
y=10,\quad \hat{y}=12
$$

has:

$$
10-12=-2
$$

If we simply averaged errors:

$$
\frac{2+(-2)}{2}=0
$$

which would incorrectly suggest no error.

Squaring gives:

$$
2^2=4
$$

and:

$$
(-2)^2=4
$$

so the errors do not cancel.

---

# 11. Why We Need Gradients

Our model contains parameters:

$$
w_0,w_1,\ldots,w_n
$$

Initially, we do not know the correct values.

We need an optimization method to find parameter values that minimize the loss.

The loss function can be thought of as a surface over the parameter space.

The gradient tells us the direction in which the loss increases most rapidly.

Therefore, to reduce the loss, we move in the opposite direction of the gradient.

This is the basic idea behind Gradient Descent.

---

# 12. Deriving the Gradient

Consider:

$$
\hat{y}_i=w_0+w_1x_i+w_2x_i^2+\cdots+w_nx_i^n
$$

The MSE is:

$$
L=
\frac{1}{m}
\sum_{i=1}^{m}
(y_i-\hat{y}_i)^2
$$

For convenience, define the error:

$$
e_i=\hat{y}_i-y_i
$$

Then:

$$
L=
\frac{1}{m}
\sum_{i=1}^{m}e_i^2
$$

## Gradient with respect to a weight

For a particular weight \(w_j\):

$$
\frac{\partial L}{\partial w_j}
=
\frac{2}{m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)x_i^j
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w_j}
=
\frac{2}{m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)x_i^j
}
$$

This is the central gradient equation for each polynomial weight.

## Gradient with respect to the bias

Since the bias has no \(x\) term:

$$
\boxed{
\frac{\partial L}{\partial w_0}
=
\frac{2}{m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)
}
$$

If we represent the intercept as a parameter \(w_0\), this is simply the \(j=0\) case because:

$$
x^0=1
$$

---

# 13. Gradient Descent

Once we calculate the gradients, update each parameter.

For a weight:

$$
\boxed{
w_j \leftarrow
w_j-\alpha
\frac{\partial L}{\partial w_j}
}
$$

For the bias:

$$
\boxed{
w_0 \leftarrow
w_0-\alpha
\frac{\partial L}{\partial w_0}
}
$$

where:

$$
\alpha=\text{learning rate}
$$

The learning rate controls how large each update is.

### Learning rate too small

Training may be very slow.

### Learning rate too large

The algorithm may overshoot the minimum or become unstable.

### Reasonable learning rate

The loss should generally decrease toward a minimum.

---

# 14. The Complete Training Process

The complete algorithm is iterative.

```text
Initialize weights and bias
          ↓
Create polynomial features
          ↓
Calculate predictions
          ↓
Calculate loss
          ↓
Calculate gradients
          ↓
Update weights and bias
          ↓
Repeat for many iterations
          ↓
Return trained parameters
```

Mathematically:

### Step 1 — Transform features

$$
X \rightarrow X_{poly}
$$

### Step 2 — Initialize parameters

$$
W=0,\quad b=0
$$

### Step 3 — Predict

$$
\hat{Y}=X_{poly}W+b
$$

### Step 4 — Calculate error

$$
E=\hat{Y}-Y
$$

### Step 5 — Calculate gradients

$$
dW=
\frac{2}{m}X_{poly}^TE
$$

$$
db=
\frac{2}{m}\sum E
$$

### Step 6 — Update parameters

$$
W\leftarrow W-\alpha dW
$$

$$
b\leftarrow b-\alpha db
$$

### Step 7 — Repeat

Repeat for the chosen number of iterations.

---

# 15. Closed-Form Solution

Polynomial Regression can also be solved without Gradient Descent using the Normal Equation.

For:

$$
Y=X\theta
$$

the least-squares solution is:

$$
\boxed{
\theta=(X^TX)^{-1}X^TY
}
$$

However, directly calculating:

$$
(X^TX)^{-1}
$$

can be numerically problematic and computationally expensive for large feature matrices.

A more numerically stable approach uses the pseudoinverse:

$$
\boxed{
\theta=X^+Y
}
$$

where \(X^+\) is the Moore-Penrose pseudoinverse.

In NumPy:

```python
theta = np.linalg.pinv(X_poly) @ Y
```

This is useful for understanding the closed-form solution, but our main implementation can use Gradient Descent so that the optimization process is explicit.

---

# 16. Why Use Gradient Descent in a From-Scratch Project?

Using Gradient Descent makes several important ML concepts visible:

- Loss functions
- Partial derivatives
- Gradients
- Learning rate
- Iterative optimization
- Parameter updates
- Convergence

It also connects directly to the optimization process used throughout machine learning.

The objective is not merely to obtain a polynomial curve. It is to understand how the parameters are learned.

---

# 17. Multiple Input Features

Polynomial Regression becomes more interesting when there are multiple input features.

Suppose we have:

$$
x_1,\ x_2
$$

For degree 2, polynomial expansion can include:

$$
x_1,\quad x_2,\quad x_1^2,\quad x_2^2,\quad x_1x_2
$$

The model can therefore be:

$$
\hat{y}
=
w_0+
w_1x_1+
w_2x_2+
w_3x_1^2+
w_4x_2^2+
w_5x_1x_2
$$

The cross-term:

$$
x_1x_2
$$

is called an **interaction term**.

This allows the model to represent relationships where the effect of one feature depends on another feature.

---

# 18. Polynomial Degree

The polynomial degree controls the flexibility of the model.

### Degree 1

$$
\hat{y}=w_0+w_1x
$$

This is ordinary Linear Regression.

### Degree 2

$$
\hat{y}=w_0+w_1x+w_2x^2
$$

Can represent a basic curve.

### Degree 3

$$
\hat{y}=w_0+w_1x+w_2x^2+w_3x^3
$$

Can represent more complex curvature.

### Higher degrees

$$
\hat{y}=w_0+w_1x+\cdots+w_nx^n
$$

provide increasingly flexible functions.

But flexibility is not always beneficial.

---

# 19. Underfitting and Overfitting

## Underfitting

A model underfits when it is too simple to capture the underlying structure.

Example:

```text
Actual relationship: curved
Model: straight line
```

Typical characteristics:

- High training error
- High validation/test error
- Model lacks sufficient complexity

---

## Overfitting

A model overfits when it becomes excessively complex and starts fitting noise rather than the underlying relationship.

For example:

```text
Low-degree polynomial → smooth curve
High-degree polynomial → extremely wavy curve
```

A high-degree polynomial can pass very close to almost every training observation while performing poorly on unseen data.

Typical characteristics:

- Very low training error
- Higher validation/test error
- Excessive model complexity

---

# 20. Bias and Variance

Polynomial degree affects the bias-variance trade-off.

### Low degree

Usually:

- Higher bias
- Lower variance
- Less flexibility

### High degree

Usually:

- Lower bias
- Higher variance
- More flexibility

The goal is not to choose the highest possible degree.

The goal is to find a model complexity that generalizes well to unseen data.

---

# 21. Feature Scaling

Polynomial features can grow extremely quickly.

Suppose:

$$
x=100
$$

Then:

$$
x^2=10,000
$$

$$
x^3=1,000,000
$$

$$
x^4=100,000,000
$$

Large feature magnitudes can make Gradient Descent difficult to optimize.

For example, the gradient associated with \(x^4\) can be much larger than the gradient associated with \(x\).

This can make optimization unstable or force us to use an extremely small learning rate.

---

## Standardization

A common approach is standardization:

$$
\boxed{
z=\frac{x-\mu}{\sigma}
}
$$

where:

- \(\mu\) = mean of the feature
- \(\sigma\) = standard deviation

Polynomial features can then be generated from the scaled feature.

For example:

$$
z,\ z^2,\ z^3
$$

instead of:

$$
x,\ x^2,\ x^3
$$

Feature scaling is especially important for higher-degree polynomial models trained using Gradient Descent.

---

# 22. Regularization

Increasing polynomial degree increases model flexibility.

Regularization adds a penalty to discourage excessively large parameter values.

For example, Ridge Regression adds an L2 penalty:

$$
\boxed{
L=
MSE+\lambda\sum_{j=1}^{n}w_j^2
}
$$

where:

$$
\lambda \geq 0
$$

controls the strength of regularization.

Larger \(\lambda\) generally produces stronger parameter shrinkage.

Regularization can help reduce overfitting.

---

# 23. Ridge and Lasso with Polynomial Features

Polynomial feature generation and regularization are separate concepts.

You can combine them:

```text
Original features
      ↓
Polynomial transformation
      ↓
Polynomial features
      ↓
Ridge / Lasso
      ↓
Prediction
```

### Polynomial + Ridge

Uses:

$$
MSE+\lambda\sum w_j^2
$$

### Polynomial + Lasso

Uses:

$$
MSE+\lambda\sum |w_j|
$$

Ridge tends to shrink coefficients toward zero, while Lasso can drive some coefficients exactly to zero.

This is useful when polynomial expansion creates many features.

---

# 24. Model Evaluation

Training loss alone is not enough.

A model can have a very low training loss and still generalize poorly.

Evaluation should therefore consider predictions on data that was not used to fit the model.

Common regression metrics include:

- MAE
- MSE
- RMSE
- \(R^2\)
- Adjusted \(R^2\)

---

# 25. R² Score

The coefficient of determination is:

$$
\boxed{
R^2=
1-
\frac{\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{m}(y_i-\bar{y})^2}
}
$$

where:

$$
\bar{y}=\frac{1}{m}\sum_{i=1}^{m}y_i
$$

The numerator is the **Residual Sum of Squares**:

$$
RSS=
\sum(y_i-\hat{y}_i)^2
$$

The denominator is the **Total Sum of Squares**:

$$
TSS=
\sum(y_i-\bar{y})^2
$$

Therefore:

$$
\boxed{
R^2=1-\frac{RSS}{TSS}
}
$$

### Interpretation

An \(R^2\) of:

$$
1
$$

means the predictions match the observed target values perfectly on that evaluated dataset.

An \(R^2\) close to:

$$
0
$$

means the model explains little of the target variation relative to the mean baseline.

Importantly, \(R^2\) can be negative on unseen data when the model performs worse than simply predicting the training/evaluation mean.

So \(R^2\) should not automatically be interpreted as a probability or percentage.

---

# 26. Adjusted R²

Ordinary \(R^2\) can increase when additional predictors are introduced, even when those predictors provide little useful information.

Adjusted \(R^2\) accounts for the number of predictors:

$$
\boxed{
\bar{R}^2=
1-
(1-R^2)
\frac{m-1}{m-p-1}
}
$$

where:

- \(m\) = number of observations
- \(p\) = number of predictors

Polynomial expansion can create many predictors, so adjusted \(R^2\) can be useful when comparing models with different numbers of features.

---

# 27. MAE, MSE, and RMSE

## Mean Absolute Error

$$
\boxed{
MAE=
\frac{1}{m}
\sum_{i=1}^{m}|y_i-\hat{y}_i|
}
$$

MAE represents the average absolute prediction error.

---

## Mean Squared Error

$$
\boxed{
MSE=
\frac{1}{m}
\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
}
$$

MSE penalizes larger errors more strongly because errors are squared.

---

## Root Mean Squared Error

$$
\boxed{
RMSE=\sqrt{MSE}
}
$$

RMSE has the same units as the target variable.

For example, if the target is salary in rupees, RMSE is also expressed in rupees.

---

# 28. Training, Validation, and Test Sets

A robust machine learning workflow separates the data.

### Training set

Used to learn:

$$
W,b
$$

### Validation set

Used for decisions such as:

- Polynomial degree
- Learning rate
- Number of iterations
- Regularization strength

### Test set

Used for the final evaluation after model decisions have been made.

A common conceptual split is:

```text
Dataset
   │
   ├── Training
   │
   ├── Validation
   │
   └── Test
```

For a small educational project, a train/test split is often sufficient initially. As the project becomes more rigorous, validation or cross-validation can be introduced.

---

# 29. Polynomial Regression vs Linear Regression

| Property | Linear Regression | Polynomial Regression |
|---|---|---|
| Basic form | \(w_0+w_1x\) | \(w_0+w_1x+\cdots+w_nx^n\) |
| Degree | 1 | \(n\) |
| Relationship with original \(x\) | Linear | Potentially nonlinear |
| Parameters | Weights + bias | Weights + bias |
| Can model curvature? | No | Yes |
| Optimization | Gradient Descent / closed form | Gradient Descent / closed form |
| Overfitting risk | Lower for simple models | Can increase with degree |
| Feature engineering | Minimal | Polynomial expansion |

The key distinction is that Polynomial Regression changes the feature representation.

---

# 30. Polynomial Regression vs Other Nonlinear Models

Polynomial Regression is not the only way to model nonlinear relationships.

Other approaches include:

- Decision Trees
- Random Forests
- Gradient Boosting
- Support Vector Regression with nonlinear kernels
- Neural Networks
- Splines
- Generalized Additive Models

Polynomial Regression is especially useful when the relationship can be reasonably approximated by a smooth polynomial.

It may be less appropriate when the true relationship contains discontinuities, sharp local behavior, or complex structures that require much greater flexibility.

---

# 31. Important Assumptions

Polynomial Regression inherits many of the assumptions associated with ordinary least-squares regression.

Depending on the statistical objective, important considerations include:

### 1. Appropriate functional form

The polynomial representation should reasonably describe the relationship between predictors and target.

### 2. Independent observations

Observations should be appropriately independent for the intended statistical interpretation.

### 3. Homoscedasticity

The variance of the residuals should be reasonably constant across the range of fitted values when classical inference is being used.

### 4. Residual behavior

For statistical inference, residual distributions and dependence should be examined.

### 5. Multicollinearity

Polynomial features such as:

$$
x,\ x^2,\ x^3
$$

can be highly correlated, especially when \(x\) is not centered or scaled.

This can make parameter estimates unstable.

### 6. Outliers

Polynomial models can be strongly affected by extreme observations.

A single outlier can significantly change the fitted curve.

---

# 32. Common Mistakes

## Mistake 1 — Thinking Polynomial Regression is completely different from Linear Regression

The optimization can remain essentially the same.

The major difference is the feature representation:

$$
X\rightarrow X_{poly}
$$

---

## Mistake 2 — Forgetting the bias

A model such as:

$$
\hat{y}=w_1x+w_2x^2
$$

cannot represent the same set of functions as:

$$
\hat{y}=w_0+w_1x+w_2x^2
$$

unless the intercept is intentionally constrained to zero.

---

## Mistake 3 — Using a very high degree immediately

A high-degree polynomial can memorize training data and produce unstable predictions.

Start with:

$$
degree=2
$$

then experiment with:

$$
3,\ 4,\ 5,\ldots
$$

and evaluate on unseen data.

---

## Mistake 4 — Ignoring feature scaling

High-degree terms can become enormous.

Always inspect feature magnitudes before applying Gradient Descent.

---

## Mistake 5 — Evaluating only on training data

A high training \(R^2\) does not prove good generalization.

Use a separate evaluation dataset.

---

## Mistake 6 — Using Scikit-learn for the core algorithm

For this project, avoid:

```python
PolynomialFeatures
LinearRegression
```

from Scikit-learn.

They hide exactly the operations we want to understand.

---

# 33. From Mathematics to NumPy

The mathematical model:

$$
\hat{Y}=X_{poly}W+b
$$

maps naturally to NumPy.

### Polynomial features

Mathematics:

$$
X_{poly}=[X,X^2,X^3,\ldots,X^n]
$$

Conceptually:

```python
features = [X ** power for power in range(1, degree + 1)]
X_poly = np.column_stack(features)
```

### Prediction

Mathematics:

$$
\hat{Y}=X_{poly}W+b
$$

NumPy:

```python
Y_pred = X_poly @ weights + bias
```

The `@` operator performs matrix multiplication.

---

## Loss

Mathematics:

$$
MSE=
\frac{1}{m}
\sum(Y-\hat{Y})^2
$$

NumPy:

```python
error = Y_pred - Y
loss = np.mean(error ** 2)
```

---

## Gradient of the weights

Mathematics:

$$
dW=
\frac{2}{m}X_{poly}^T(Y_{pred}-Y)
$$

NumPy:

```python
dw = (2 / m) * (X_poly.T @ error)
```

---

## Gradient of the bias

Mathematics:

$$
db=
\frac{2}{m}\sum(Y_{pred}-Y)
$$

NumPy:

```python
db = (2 / m) * np.sum(error)
```

---

## Parameter update

Mathematics:

$$
W\leftarrow W-\alpha dW
$$

$$
b\leftarrow b-\alpha db
$$

NumPy:

```python
weights -= learning_rate * dw
bias -= learning_rate * db
```

This is a good example of how mathematical notation maps directly into vectorized numerical code.

---

# 34. Recommended Class Design

A simple implementation can be organized as:

```python
class PolynomialRegression:

    def __init__(
        self,
        degree=2,
        learning_rate=0.01,
        iterations=1000
    ):
        ...

    def create_polynomial_features(self, X):
        ...

    def predict(self, X):
        ...

    def calculate_loss(self, Y, Y_pred):
        ...

    def calculate_gradients(self, X, Y, Y_pred):
        ...

    def fit(self, X, Y):
        ...

    def r2_score(self, Y, Y_pred):
        ...
```

The methods have distinct responsibilities.

| Method | Responsibility |
|---|---|
| `__init__` | Store hyperparameters and initialize model configuration |
| `create_polynomial_features()` | Generate \(X,X^2,\ldots,X^n\) |
| `predict()` | Calculate model predictions |
| `calculate_loss()` | Calculate MSE |
| `calculate_gradients()` | Calculate \(dW\) and \(db\) |
| `fit()` | Train the model |
| `r2_score()` | Evaluate explained variance |

A clean implementation keeps feature engineering, prediction, optimization, and evaluation conceptually separate.

---

# 35. Complete Conceptual Pipeline

The entire project can be understood as:

```text
                 RAW DATA
                    │
                    ▼
             Input Feature X
                    │
                    ▼
       Polynomial Feature Expansion
                    │
                    ▼
        ┌─────────────────────────┐
        │ X, X², X³, ..., Xⁿ      │
        └─────────────────────────┘
                    │
                    ▼
           Initialize W and b
                    │
                    ▼
                Predict
                    │
                    ▼
              Calculate MSE
                    │
                    ▼
            Calculate Gradients
                    │
                    ▼
           Gradient Descent
                    │
                    ▼
             Update W and b
                    │
                    ▼
             Repeat Training
                    │
                    ▼
              Trained Model
                    │
                    ▼
               Predictions
                    │
                    ▼
            Model Evaluation
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       R² Score          Error Metrics
          │
          ▼
       Visualization
```

---

# 36. Practical Experiment Plan

The best way to understand Polynomial Regression is to experiment with it.

## Experiment 1 — Linear relationship

Generate data approximately following:

$$
y=3x+5+\epsilon
$$

Train:

- Degree 1
- Degree 2
- Degree 3

Observe whether higher degrees provide meaningful improvement.

---

## Experiment 2 — Quadratic relationship

Generate:

$$
y=2x^2+3x+5+\epsilon
$$

Compare:

```text
Degree 1
Degree 2
Degree 3
Degree 5
```

Observe the fitted curves and evaluation metrics.

---

## Experiment 3 — Cubic relationship

Generate:

$$
y=x^3-2x^2+3x+5+\epsilon
$$

Again compare different degrees.

---

## Experiment 4 — Overfitting

Use a small noisy dataset and increase the polynomial degree:

```text
1
2
3
5
10
15
20
```

Track:

- Training MSE
- Test MSE
- Training \(R^2\)
- Test \(R^2\)

Plot the resulting curves.

This experiment makes overfitting much easier to understand than reading about it theoretically.

---

## Experiment 5 — Learning rate

Try:

```text
0.0001
0.001
0.01
0.1
```

Plot loss against iterations.

Observe:

- Slow convergence
- Stable convergence
- Fast convergence
- Divergence/instability

---

## Experiment 6 — Feature scaling

Train the same model:

1. Without scaling
2. With standardized input

Compare the training behavior.

This demonstrates why feature scaling matters for Gradient Descent.

---

# 37. Key Takeaways

### 1. Polynomial Regression extends Linear Regression

It adds powers of the input:

$$
x,x^2,x^3,\ldots,x^n
$$

---

### 2. Feature transformation is the key idea

The original input is transformed:

$$
X\rightarrow X_{poly}
$$

and a linear model is fitted to the transformed features.

---

### 3. The model is nonlinear in the original input

For example:

$$
\hat{y}=w_0+w_1x+w_2x^2
$$

produces a curved relationship between \(x\) and \(\hat y\).

---

### 4. The model is linear in its parameters

The weights:

$$
w_0,w_1,w_2
$$

appear linearly.

This is why Polynomial Regression can be treated as a form of Multiple Linear Regression after feature transformation.

---

### 5. Gradient Descent learns the parameters

The model repeatedly:

```text
Predict
  ↓
Calculate error
  ↓
Calculate gradients
  ↓
Update parameters
  ↓
Repeat
```

---

### 6. Degree controls model complexity

Low degree:

```text
Simpler model
```

High degree:

```text
More flexible model
```

Too much flexibility can cause overfitting.

---

### 7. Feature scaling matters

Polynomial powers can grow rapidly, making optimization difficult.

---

### 8. Evaluation must consider unseen data

A model that fits training data extremely well may still generalize poorly.

---

# 38. Further Topics

Once the basic implementation is working, the following topics are natural extensions:

- Polynomial Regression with multiple features
- Interaction terms
- Standardization
- Normalization
- Ridge Regression
- Lasso Regression
- Elastic Net
- Cross-validation
- Learning curves
- Bias-variance analysis
- Numerical stability
- Condition numbers
- Multicollinearity
- Vandermonde matrices
- Normal Equation
- Moore-Penrose pseudoinverse
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-batch Gradient Descent
- Hyperparameter tuning
- Polynomial splines
- Basis functions

---

# Final Mental Model

If you remember only one pipeline from this document, remember this:

$$
\boxed{
X
\rightarrow
$$
X,X^2,\ldots,X^n
$$
\rightarrow
\text{Linear Model}
\rightarrow
\hat{Y}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradients}
\rightarrow
\text{Parameter Updates}
}
$$

Polynomial Regression is therefore not about creating an entirely new optimization algorithm.

The central idea is:

> **Transform the input features into polynomial features, then learn a linear combination of those transformed features.**

That single idea connects the mathematics, NumPy implementation, Gradient Descent, model complexity, and practical behavior of Polynomial Regression.
