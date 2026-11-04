# Machine Learning - Practical Session 3

Hypothesis spaces, feature representations and model generalization

---

## Part A - Exploring different hypothesis spaces

**1. Dataset**
- 10 observations, 2 variables (`Study Hours`, `Exam Score`): shape `(10, 3)` with the student id.
- Study hours: mean 5.5, std 3.03, range 1 to 10. Exam score: mean 67.9, std 16.91, range 42 to 91.
- The scatter plot shows an almost perfectly linear, increasing relationship: about +5.3 points per extra hour.

**3. Training performance**

| Model | Training RMSE | Training R² |
|---|---|---|
| Linear Regression | 1.1672 | 0.9947 |
| Polynomial Regression (degree 2) | 1.0435 | 0.9958 |
| Polynomial Regression (degree 3) | 1.0211 | 0.9959 |
| Decision Tree (depth 1) | 8.0050 | 0.7509 |
| Decision Tree (depth 3) | 1.1180 | 0.9951 |
| KNN (k = 3) | 2.6562 | 0.9726 |

**Comparison of the fitted curves**
- **Linear regression**: a straight line. It already fits very well because the data is almost linear.
- **Polynomial degree 2 and 3**: curves almost identical to the line. Their training RMSE is only slightly lower. A more flexible model always fits the training data at least as well as a simpler one it contains, so this small gain does not prove it is better.
- **Decision tree depth 1**: one split, so only two constant values (a single step). It cannot follow the trend: it **underfits** (RMSE 8.0, R² 0.75).
- **Decision tree depth 3**: up to 8 steps for 10 points, so it nearly memorizes the data (RMSE 1.12). Its prediction is a staircase, not a smooth trend.
- **KNN (k = 3)**: averages the 3 nearest students, so it is smoother than a tree but flattens at both ends (RMSE 2.66).
- Each model family represents the relationship differently: global line or curve (linear, polynomial) versus local constants (trees, KNN). The hypothesis space decides which shapes can be learned.

---

## Part B - Model complexity and generalization

**1. Leave-One-Out**
LOOCV does 10 iterations. In each one, 1 student is the test set and the other 9 are the training set. The test point moves from S1 (iteration 1) to S10 (iteration 10), and the training set always contains the 9 others. Every observation is used exactly once for testing.

**2-4. Results**

| Polynomial Degree | Training RMSE | LOOCV RMSE |
|---|---|---|
| 1 | 1.1672 | 1.5486 |
| 2 | 1.0435 | 1.5728 |
| 3 | 1.0211 | 2.0620 |
| 5 | 0.8977 | 3.5670 |
| 9 | 2.7957 | 34.3697 |

**Analysis**
- Training RMSE decreases from degree 1 to degree 5 (1.17 to 0.90), but LOOCV RMSE **increases** (1.55 to 3.57). The more complex model fits the training points better and predicts unseen points worse. This is **overfitting**.
- The model with the lowest LOOCV RMSE is **degree 1** (1.5486). Degree 2 is almost the same (1.5728).
- Degree 9 is the worst: LOOCV RMSE 34.37. With 9 training points in each fold, a degree-9 polynomial has more parameters than data and oscillates wildly, especially when it must predict the first or last point (extrapolation).
- **Note on degree 9 training RMSE (2.7957):** in theory a degree-9 polynomial can pass through all 10 points (RMSE = 0). The value 2.80 is a numerical artifact: the raw features go up to 10⁹, and the design matrix is ill-conditioned (condition number about 2×10¹²). When I tested the same model with feature scaling, the training RMSE dropped to 0.67. This is why scaling inside a Pipeline matters, and why Parts C and D use it for Ridge and Lasso.
- Conclusion: low training error does not mean good generalization. LOOCV error is the honest measure.

---

## Part C - Feature representation and regularization

**1. Feature representations (LOOCV RMSE)**

| Representation | LOOCV RMSE |
|---|---|
| Original (x) | 1.5486 |
| Polynomial (x, x², x³) | 2.0620 |
| Higher-degree (x ... x⁵) | 3.5670 |

The original representation is the best. Adding polynomial features gives the model more flexibility than the data needs, and with 10 points it uses that flexibility to fit noise.

**2. Regularization (degree-5 features, Pipeline: PolynomialFeatures, StandardScaler, model)**

| alpha | Ridge LOOCV RMSE | Lasso LOOCV RMSE |
|---|---|---|
| 0.01 | 2.1442 | 1.7654 |
| 0.1 | **1.8071** | **1.6223** |
| 1 | 3.5735 | 1.9187 |
| 10 | 7.2820 | 11.8773 |

- Best alpha is 0.1 for both.
- **Too small** (0.01): almost no regularization, so the model still overfits.
- **Too large** (10): the model is over-constrained and underfits (predicts close to the mean), so error rises again.
- **Ridge (L2)** shrinks all coefficients but keeps them non-zero: [12.92, 4.99, 1.09, -1.01, -2.43].
- **Lasso (L1)** sets coefficients exactly to zero: [16.29, 0, 0, 0, -0.47]. It kept x and x⁵ and removed x², x³, x⁴, so it also does feature selection. (Coefficients are on standardized features.)

**3. Comparison table**

| Model | Feature Degree | Regularization | LOOCV RMSE |
|---|---|---|---|
| Linear Regression | 1 | None | 1.5486 |
| Polynomial Regression | 3 | None | 2.0620 |
| Polynomial Regression | 5 | None | 3.5670 |
| Ridge Regression (alpha = 0.1) | 5 | L2 | 1.8071 |
| Lasso Regression (alpha = 0.1) | 5 | L1 | 1.6223 |
| Decision Tree depth 1 | - | None | 13.4583 |
| Decision Tree depth 3 | - | None | 6.3659 |
| KNN k = 3 | - | None | 6.0037 |

**Does regularization help?** Yes, for degree-5 polynomial regression: LOOCV RMSE falls from 3.5670 (no regularization) to 1.8071 (Ridge) and 1.6223 (Lasso). It does not beat plain linear regression (1.5486), because the true relationship is already linear and there is nothing for extra flexibility to capture. Regularization repairs the damage of an unnecessarily complex model, but does not replace choosing an appropriate one.

**Caveat:** alpha was chosen using the same LOOCV that reports the score, so Ridge and Lasso numbers are slightly optimistic. A nested cross-validation would remove this bias.

**Trees and KNN** have the worst LOOCV errors (13.46, 6.37, 6.00). Most likely reason: they predict constants from nearby training points, so they cannot extrapolate. When the held-out student is at one end of the range (1 or 10 hours), they cannot continue the upward trend, whereas a linear model can.

---

## Part D - Final model comparison

Best configuration of each family (by LOOCV from Parts A, B, C):

| Model | Training RMSE | LOOCV RMSE | Gap (LOOCV - Train) |
|---|---|---|---|
| Linear Regression | 1.1672 | **1.5486** | **0.38** |
| Polynomial (degree 2) | 1.0435 | 1.5728 | 0.53 |
| Ridge (degree 5, alpha = 0.1) | 1.0865 | 1.8071 | 0.72 |
| Lasso (degree 5, alpha = 0.1) | 1.0656 | 1.6223 | 0.56 |
| Decision Tree (depth 3) | 1.1180 | 6.3659 | 5.25 |
| KNN (k = 3) | 2.6562 | 6.0037 | 3.35 |

Polynomial degree 2 is the best polynomial degree among 2, 3, 5 and 9. Tree depth 3 is better than depth 1.

**Comparison of predictions with observations** (figure `fig_D_final.png`): the linear, polynomial, Ridge and Lasso curves all follow the data as a smooth upward line. The tree is a staircase and KNN is a flattened curve.

**Best balance between fitting and generalizing: Linear Regression.** It has the lowest LOOCV RMSE (1.5486) and the smallest gap between training and LOOCV error (0.38), and it is the simplest model. Polynomial degree 2 and Lasso are very close. With only 10 observations, the differences among these three (1.55 to 1.62) are small and should not be treated as significant. The clear conclusion is the large difference between the linear-type models and the tree/KNN models.

---

## Conclusion

- **Model complexity:** training error always improves with complexity, but generalization does not. LOOCV RMSE rose from 1.55 (degree 1) to 34.37 (degree 9). The right complexity is the one that matches the data: here, a straight line.
- **Feature representation:** the original feature x was better than x, x², x³ (2.06) or x to x⁵ (3.57). More features only help if they carry information the model needs.
- **Regularization:** Ridge and Lasso reduced the overfitting of the degree-5 model (3.57 to 1.81 and 1.62), and Lasso also removed useless features. The strength alpha must be tuned: too small gives overfitting, too large gives underfitting.
- **Hypothesis space:** models that extrapolate a global trend (linear, polynomial) did far better here than local-constant models (trees, KNN), because the data is almost linear.
- **Method:** all models were evaluated on the same data with the same LOOCV, and scaling was done inside a Pipeline to avoid data leakage.
