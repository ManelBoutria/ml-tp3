import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

D = pd.DataFrame({
    "Student": [f"S{i}" for i in range(1, 11)],
    "Study Hours": list(range(1, 11)),
    "Exam Score": [42, 48, 55, 59, 66, 70, 78, 81, 89, 91],
})
X = D[["Study Hours"]].values
y = D["Exam Score"].values
xs = np.linspace(1, 10, 300).reshape(-1, 1)
loo = LeaveOneOut()


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def train_rmse(model):
    model.fit(X, y)
    return rmse(y, model.predict(X))


def loocv_rmse(model):
    s = cross_val_score(model, X, y, cv=loo, scoring="neg_mean_squared_error")
    return np.sqrt(-s.mean())


def poly_model(d):
    return make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression())


def ridge_model(a, d=5):
    return make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), Ridge(alpha=a))


def lasso_model(a, d=5):
    return make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(),
                         Lasso(alpha=a, max_iter=200000))


print("===== PART A =====")

print("A1")
print(D)
print(D.shape)
print(D.describe())
plt.figure(figsize=(6, 4))
plt.scatter(D["Study Hours"], D["Exam Score"], color="black")
plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Study hours vs exam score")
plt.grid(alpha=0.3)
plt.savefig("fig_A1_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

print("A2-A3")
models_A = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (degree 2)": poly_model(2),
    "Polynomial Regression (degree 3)": poly_model(3),
    "Decision Tree (depth 1)": DecisionTreeRegressor(max_depth=1, random_state=42),
    "Decision Tree (depth 3)": DecisionTreeRegressor(max_depth=3, random_state=42),
    "KNN (k = 3)": KNeighborsRegressor(n_neighbors=3),
}
rows = []
plt.figure(figsize=(9, 6))
plt.scatter(X, y, color="black", zorder=5, label="Observations")
for name, m in models_A.items():
    m.fit(X, y)
    pred = m.predict(X)
    rows.append([name, rmse(y, pred), r2_score(y, pred)])
    plt.plot(xs, m.predict(xs), label=name)
plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Fitted curves of the six models")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.savefig("fig_A3_models.png", dpi=150, bbox_inches="tight")
plt.show()
table_A = pd.DataFrame(rows, columns=["Model", "Training RMSE", "Training R2"])
print(table_A.to_string(index=False))

print("===== PART B =====")

print("B1")
print(loo.get_n_splits(X))
for i, (tr, te) in enumerate(loo.split(X), start=1):
    print(f"Iteration {i}: train indices = {tr.tolist()} | test index = {te.tolist()}")
    print("   train hours =", X[tr].ravel().tolist(), "| test hours =", X[te].ravel().tolist())

print("B2-B4")
degrees = [1, 2, 3, 5, 9]
rows = []
for d in degrees:
    rows.append([d, train_rmse(poly_model(d)), loocv_rmse(poly_model(d))])
table_B = pd.DataFrame(rows, columns=["Polynomial Degree", "Training RMSE", "LOOCV RMSE"])
print(table_B.to_string(index=False))
print("Best degree by LOOCV:", int(table_B.loc[table_B["LOOCV RMSE"].idxmin(), "Polynomial Degree"]))

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(table_B["Polynomial Degree"], table_B["Training RMSE"], "o-", label="Training RMSE")
plt.plot(table_B["Polynomial Degree"], table_B["LOOCV RMSE"], "s-", label="LOOCV RMSE")
plt.yscale("log")
plt.xlabel("Polynomial degree")
plt.ylabel("RMSE (log scale)")
plt.title("Training vs LOOCV RMSE")
plt.legend()
plt.grid(alpha=0.3)
plt.subplot(1, 2, 2)
plt.scatter(X, y, color="black", zorder=5)
for d in degrees:
    m = poly_model(d).fit(X, y)
    plt.plot(xs, m.predict(xs), label=f"degree {d}")
plt.ylim(30, 105)
plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Polynomial fits")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_B_degrees.png", dpi=150, bbox_inches="tight")
plt.show()

print("===== PART C =====")

print("C1")
reps = {
    "Original (x)": LinearRegression(),
    "Polynomial (x, x2, x3)": poly_model(3),
    "Polynomial (x ... x5)": poly_model(5),
}
for name, m in reps.items():
    print(f"{name:28s} LOOCV RMSE = {loocv_rmse(m):.4f}")

print("C2")
alphas = [0.01, 0.1, 1, 10]
rows = []
for a in alphas:
    rows.append([a, loocv_rmse(ridge_model(a)), loocv_rmse(lasso_model(a))])
table_C = pd.DataFrame(rows, columns=["alpha", "Ridge LOOCV RMSE", "Lasso LOOCV RMSE"])
print(table_C.to_string(index=False))
best_a_ridge = alphas[int(np.argmin(table_C["Ridge LOOCV RMSE"]))]
best_a_lasso = alphas[int(np.argmin(table_C["Lasso LOOCV RMSE"]))]
print("Best alpha Ridge:", best_a_ridge, "| Best alpha Lasso:", best_a_lasso)
lasso_best = lasso_model(best_a_lasso).fit(X, y)
print("Lasso coefficients (alpha =", best_a_lasso, "):", np.round(lasso_best[-1].coef_, 3))
ridge_best = ridge_model(best_a_ridge).fit(X, y)
print("Ridge coefficients (alpha =", best_a_ridge, "):", np.round(ridge_best[-1].coef_, 3))

print("C3")
rows = [
    ["Linear Regression", 1, "None", loocv_rmse(LinearRegression())],
    ["Polynomial Regression", 3, "None", loocv_rmse(poly_model(3))],
    ["Polynomial Regression", 5, "None", loocv_rmse(poly_model(5))],
    [f"Ridge Regression (alpha={best_a_ridge})", 5, "L2", loocv_rmse(ridge_model(best_a_ridge))],
    [f"Lasso Regression (alpha={best_a_lasso})", 5, "L1", loocv_rmse(lasso_model(best_a_lasso))],
    ["Decision Tree depth 1", "-", "None", loocv_rmse(DecisionTreeRegressor(max_depth=1, random_state=42))],
    ["Decision Tree depth 3", "-", "None", loocv_rmse(DecisionTreeRegressor(max_depth=3, random_state=42))],
    ["KNN k = 3", "-", "None", loocv_rmse(KNeighborsRegressor(n_neighbors=3))],
]
table_C3 = pd.DataFrame(rows, columns=["Model", "Feature Degree", "Regularization", "LOOCV RMSE"])
print(table_C3.to_string(index=False))

print("===== PART D =====")

best_deg = min([2, 3, 5, 9], key=lambda d: loocv_rmse(poly_model(d)))
best_depth = min([1, 3], key=lambda k: loocv_rmse(DecisionTreeRegressor(max_depth=k, random_state=42)))
final_models = {
    "Linear Regression": LinearRegression(),
    f"Polynomial (degree {best_deg})": poly_model(best_deg),
    f"Ridge (deg 5, alpha={best_a_ridge})": ridge_model(best_a_ridge),
    f"Lasso (deg 5, alpha={best_a_lasso})": lasso_model(best_a_lasso),
    f"Decision Tree (depth {best_depth})": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "KNN (k = 3)": KNeighborsRegressor(n_neighbors=3),
}
rows = []
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color="black", zorder=5, label="Observations")
for name, m in final_models.items():
    tr = train_rmse(m)
    cv = loocv_rmse(m)
    rows.append([name, tr, cv])
    plt.plot(xs, m.predict(xs), label=name)
plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Best configuration of each family")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
table_D = pd.DataFrame(rows, columns=["Model", "Training RMSE", "LOOCV RMSE"])
table_D["Gap (LOOCV - Train)"] = table_D["LOOCV RMSE"] - table_D["Training RMSE"]
print(table_D.to_string(index=False))
print("Best LOOCV RMSE:", table_D.loc[table_D["LOOCV RMSE"].idxmin(), "Model"])
plt.subplot(1, 2, 2)
idx = np.arange(len(table_D))
plt.bar(idx - 0.2, table_D["Training RMSE"], 0.4, label="Training RMSE")
plt.bar(idx + 0.2, table_D["LOOCV RMSE"], 0.4, label="LOOCV RMSE")
plt.xticks(idx, [n.split(" (")[0] for n in table_D["Model"]], rotation=30, ha="right")
plt.ylabel("RMSE")
plt.title("Training vs LOOCV RMSE")
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("fig_D_final.png", dpi=150, bbox_inches="tight")
plt.show()