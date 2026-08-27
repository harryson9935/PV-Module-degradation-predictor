"""
train_models.py
----------------
End-to-end pipeline for the PV Module Degradation Predictor.

1. Load dataset + exploratory data analysis (EDA)
2. Feature selection / correlation analysis
3. Train-test split
4. Train Linear Regression, Random Forest, Gradient Boosting regressors
   to predict cumulative_degradation_pct
5. Hyperparameter tuning via GridSearchCV (RF, GBM)
6. Evaluate with R^2, RMSE, MAE (regression) and a derived classification
   view (fault_flag) with a confusion matrix, using the best regressor's
   predicted degradation compared against the warranty-limit threshold
7. Save all plots to /images and a results summary to /results
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")

BASE = "/home/claude/pv-degradation-predictor"
DATA_PATH = f"{BASE}/data/pv_degradation_dataset.csv"
IMG_DIR = f"{BASE}/images"
RESULTS_DIR = f"{BASE}/results"

FEATURES = [
    "module_age_years", "ambient_temp_c", "module_temp_c",
    "relative_humidity_pct", "daily_temp_swing_c",
    "irradiance_kwh_m2_day", "soiling_index",
]
TARGET = "cumulative_degradation_pct"

RANDOM_STATE = 42

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# ---------------------------------------------------------------- EDA
plt.figure(figsize=(10, 8))
corr = df[FEATURES + [TARGET, "fault_flag"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", square=True, cbar_kws={"shrink": .8})
plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/01_correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, FEATURES + [TARGET]):
    sns.histplot(df[col], kde=True, ax=ax, color="#3b7ddd")
    ax.set_title(col)
axes.flat[-1].axis("off") if len(FEATURES) + 1 < axes.size else None
plt.suptitle("Feature & Target Distributions (EDA)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/02_feature_distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="module_age_years", y=TARGET, hue="fault_flag",
                 palette={0: "#2ecc71", 1: "#e74c3c"}, alpha=0.6, s=25)
plt.title("Degradation vs Module Age (colored by fault flag)", fontsize=13, fontweight="bold")
plt.xlabel("Module Age (years)")
plt.ylabel("Cumulative Degradation (%)")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/03_degradation_vs_age.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- split
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------- models
results = {}

# --- Linear Regression (baseline) ---
lr = LinearRegression()
lr.fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)
results["Linear Regression"] = {
    "R2": r2_score(y_test, pred_lr),
    "RMSE": mean_squared_error(y_test, pred_lr) ** 0.5,
    "MAE": mean_absolute_error(y_test, pred_lr),
}

# --- Random Forest + GridSearchCV ---
rf_grid = {
    "n_estimators": [150, 300],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [1, 3],
}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    rf_grid, cv=5, scoring="r2", n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
pred_rf = best_rf.predict(X_test)
results["Random Forest"] = {
    "R2": r2_score(y_test, pred_rf),
    "RMSE": mean_squared_error(y_test, pred_rf) ** 0.5,
    "MAE": mean_absolute_error(y_test, pred_rf),
    "best_params": rf_search.best_params_,
}

# --- Gradient Boosting + GridSearchCV ---
gbm_grid = {
    "n_estimators": [150, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
}
gbm_search = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    gbm_grid, cv=5, scoring="r2", n_jobs=-1
)
gbm_search.fit(X_train, y_train)
best_gbm = gbm_search.best_estimator_
pred_gbm = best_gbm.predict(X_test)
results["Gradient Boosting"] = {
    "R2": r2_score(y_test, pred_gbm),
    "RMSE": mean_squared_error(y_test, pred_gbm) ** 0.5,
    "MAE": mean_absolute_error(y_test, pred_gbm),
    "best_params": gbm_search.best_params_,
}

# ---------------------------------------------------------------- pick best model
best_name = max(results, key=lambda k: results[k]["R2"])
best_pred = {"Linear Regression": pred_lr, "Random Forest": pred_rf,
             "Gradient Boosting": pred_gbm}[best_name]
print("\nBest model:", best_name)
for name, m in results.items():
    print(f"  {name:20s}  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

# ---------------------------------------------------------------- cross-val (best model)
cv_model = best_rf if best_name == "Random Forest" else (
    best_gbm if best_name == "Gradient Boosting" else lr)
cv_input = X_train if best_name in ("Random Forest", "Gradient Boosting") else X_train_s
cv_scores = cross_val_score(cv_model, cv_input, y_train, cv=5, scoring="r2")

# ---------------------------------------------------------------- plots: predictions
plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_pred, alpha=0.5, s=25, color="#3b7ddd")
lims = [0, max(y_test.max(), best_pred.max()) + 2]
plt.plot(lims, lims, "r--", lw=2, label="Ideal (y = x)")
plt.xlabel("Actual Cumulative Degradation (%)")
plt.ylabel("Predicted Cumulative Degradation (%)")
plt.title(f"Actual vs Predicted — {best_name}\nR² = {results[best_name]['R2']:.3f}",
          fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/04_actual_vs_predicted.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
names = list(results.keys())
r2_vals = [results[n]["R2"] for n in names]
rmse_vals = [results[n]["RMSE"] for n in names]
x_pos = np.arange(len(names))
fig, ax1 = plt.subplots(figsize=(8, 6))
bars = ax1.bar(x_pos, r2_vals, color=["#95a5a6", "#3b7ddd", "#2ecc71"])
ax1.set_ylabel("R² Score")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(names)
ax1.set_ylim(0, 1)
for bar, v in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.title("Model Comparison — R² Score", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/05_model_comparison.png", dpi=150)
plt.close()

if hasattr(best_rf, "feature_importances_"):
    fi_model, fi_label = best_rf, "Random Forest"
    if results["Gradient Boosting"]["R2"] > results["Random Forest"]["R2"]:
        fi_model, fi_label = best_gbm, "Gradient Boosting"
    importances = pd.Series(fi_model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#3b7ddd")
    plt.title(f"Feature Importance ({fi_label})", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/06_feature_importance.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------- classification view (fault detection)
warranty_limit_test = 2.5 + 0.7 * X_test["module_age_years"]
y_true_flag = (y_test.values > warranty_limit_test.values).astype(int)
y_pred_flag = (best_pred > warranty_limit_test.values).astype(int)

acc = accuracy_score(y_true_flag, y_pred_flag)
cm = confusion_matrix(y_true_flag, y_pred_flag)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Healthy", "Fault"], yticklabels=["Healthy", "Fault"])
plt.title(f"Fault Detection Confusion Matrix\nAccuracy = {acc*100:.1f}%",
          fontsize=13, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/07_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- save results summary
summary = {
    "dataset_shape": list(df.shape),
    "features": FEATURES,
    "target": TARGET,
    "regression_results": {
        k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()}
        for k, v in results.items()
    },
    "best_model": best_name,
    "best_model_r2": results[best_name]["R2"],
    "cross_val_r2_mean": float(cv_scores.mean()),
    "cross_val_r2_std": float(cv_scores.std()),
    "fault_detection_accuracy_pct": float(acc * 100),
    "confusion_matrix": cm.tolist(),
}

with open(f"{RESULTS_DIR}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

results_df = pd.DataFrame({
    name: {"R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"]}
    for name, m in results.items()
}).T
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv")

print("\n=== Fault Detection (derived classification) ===")
print(f"Accuracy: {acc*100:.2f}%")
print("Confusion matrix:\n", cm)
print(f"\n5-fold CV R2 ({best_name}): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print("\nAll results and plots saved.")
