import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

print("Downloading UCI red-wine dataset...")
df = pd.read_csv(URL, sep=";")
df.to_csv("wine_quality.csv", index=False)

print("\nRows and columns:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nOriginal quality distribution:\n", df["quality"].value_counts().sort_index())

# Remove duplicate records and create the binary target.
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)  # 1 = GOOD; 0 = NEEDS IMPROVEMENT

# Prevent target leakage: quality is not an input feature.
X = df.drop(columns=["quality"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification report:\n", classification_report(
    y_test,
    predictions,
    target_names=["Needs improvement", "Good quality"],
    zero_division=0
))

joblib.dump(model, "wine_model.pkl")
print("\nSaved wine_model.pkl")
print("Created wine_quality.csv")
