import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. LOAD DATASET
# =====================================================

file_path = (
    "/storage/emulated/0/Download/"
    "weather_prediction_3000_rows (1).csv"
)

try:
    data = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Number of rows:", len(data))
    print("Number of columns:", len(data.columns))

except Exception as e:
    print("Error loading dataset:", e)
    raise SystemExit


# =====================================================
# 2. SELECT FEATURES AND TARGET
# =====================================================

features = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "Rainfall",
    "CloudCover",
    "Pressure",
    "Visibility",
    "DewPoint",
    "UVIndex",
    "WindDirection"
]

target = "RainTomorrow"

# Check whether all required columns exist
required_columns = features + [target]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
    raise SystemExit

X = data[features].copy()

# Convert target labels into 0 and 1
y = (
    data[target]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1
    })
)

# Remove records with invalid target labels
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

# Remove rows with missing target labels or invalid labels
if len(X) == 0:
    print("Error: No valid target values were found.")
    raise SystemExit

if y.nunique() != 2:
    print("Error: RainTomorrow must contain both Yes and No.")
    raise SystemExit

print("\nFeatures selected:", len(features))
print("Valid records:", len(X))

print("\nTarget distribution:")
print(y.value_counts().rename(
    index={0: "No Rain", 1: "Rain"}
))


# =====================================================
# 3. PREPROCESSING
# =====================================================

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = [
    col for col in features
    if col not in numeric_features
]

# Fill missing numerical values with the median
numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

# Fill missing categorical values with the most
# frequent category, then encode categories
categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Apply appropriate preprocessing to each column type
preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])


# =====================================================
# 4. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# =====================================================
# 5. CREATE MACHINE LEARNING PIPELINE
# =====================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),

    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1
        )
    )
])


# =====================================================
# 6. HYPERPARAMETER TUNING
# =====================================================

print("\nTuning Random Forest model...")
print("Please wait...")

param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 15],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt"]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=1,
    refit=True
)

search.fit(X_train, y_train)

# Use the best model found during tuning
model = search.best_estimator_

print("\nTraining completed!")

print("\nBest parameters:")
print(search.best_params_)

print(
    "\nBest cross-validation ROC-AUC:",
    round(search.best_score_ * 100, 2),
    "%"
)


# =====================================================
# 7. PREDICT TEST DATA
# =====================================================

y_pred = model.predict(X_test)

# Probability that tomorrow will have rain
y_prob = model.predict_proba(X_test)[:, 1]


# =====================================================
# 8. EVALUATE MODEL PERFORMANCE
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# ROC-AUC requires both target classes in the test set
if y_test.nunique() == 2:
    roc_auc = roc_auc_score(y_test, y_prob)
else:
    roc_auc = np.nan


# =====================================================
# 9. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 45)
print("       WEATHER MODEL PERFORMANCE")
print("=" * 45)

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1-Score  :", round(f1 * 100, 2), "%")

if not np.isnan(roc_auc):
    print("ROC-AUC   :", round(roc_auc * 100, 2), "%")
else:
    print("ROC-AUC   : Not available")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Rain", "Rain"],
        zero_division=0
    )
)


# =====================================================
# 10. CONFUSION MATRIX HEATMAP
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("Weather Prediction - Confusion Matrix")
plt.colorbar()

plt.xticks(
    [0, 1],
    ["No Rain", "Rain"]
)

plt.yticks(
    [0, 1],
    ["No Rain", "Rain"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Display values inside the heatmap
threshold = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
            fontsize=14
        )

plt.tight_layout()
plt.show()


# =====================================================
# 11. FEATURE IMPORTANCE GRAPH
# =====================================================

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_forest = model.named_steps["model"]

try:
    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    importances = fitted_forest.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Show the 10 most important transformed features
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    print("\nTop 10 Important Features:")

    print(
        importance_df.to_string(
            index=False
        )
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Weather Features")

    plt.title("Top 10 Important Features")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Feature importance graph error:", e)


# =====================================================
# 12. PERFORMANCE METRICS GRAPH
# =====================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.title("Weather Prediction Performance")
plt.ylabel("Score (%)")

plt.ylim(0, 100)

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# =====================================================
# 13. ENTER NEW WEATHER DATA
# =====================================================

print("\n" + "=" * 45)
print("        ENTER NEW WEATHER DATA")
print("=" * 45)

new_values = {}

for feature in features:

    while True:

        try:
            value = input(
                "Enter " + feature + ": "
            ).strip()

            if value == "":
                print("Please enter a value.")
                continue

            if feature in numeric_features:
                new_values[feature] = float(value)
            else:
                new_values[feature] = value

            break

        except ValueError:
            print("Please enter a valid number for", feature)


# Convert input into a DataFrame with correct columns
new_data = pd.DataFrame(
    [new_values],
    columns=features
)


# =====================================================
# 14. PREDICT TOMORROW'S RAINFALL
# =====================================================

prediction = model.predict(new_data)[0]

probability = model.predict_proba(new_data)[0]

classes = list(model.named_steps["model"].classes_)

rain_index = classes.index(1)
no_rain_index = classes.index(0)

rain_probability = probability[rain_index]
no_rain_probability = probability[no_rain_index]


# =====================================================
# 15. DISPLAY FINAL PREDICTION
# =====================================================

print("\n" + "=" * 45)
print("           FINAL PREDICTION")
print("=" * 45)

if prediction == 1:
    print("Prediction: RAIN TOMORROW")
else:
    print("Prediction: NO RAIN TOMORROW")

print(
    "Rain Probability:",
    round(rain_probability * 100, 2),
    "%"
)

print(
    "No Rain Probability:",
    round(no_rain_probability * 100, 2),
    "%"
)

print("\nProgram completed successfully!")