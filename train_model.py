
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("notices_shuffled.csv")

# 2. Validate columns
required_columns = {"title", "content", "category"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "CSV must contain title, content, and category columns."
    )

# 3. Clean data
df = df.dropna(
    subset=["title", "content", "category"]
).copy()

for column in ["title", "content", "category"]:
    df[column] = df[column].astype(str).str.strip()

df = df[
    (df["title"] != "") &
    (df["content"] != "") &
    (df["category"] != "")
].copy()

# Remove exact duplicate text-category records
df["text"] = df["title"] + " " + df["content"]
df = df.drop_duplicates(
    subset=["text", "category"]
).reset_index(drop=True)

print("Dataset size:", len(df))
print("\nCategory distribution:")
print(df["category"].value_counts())

# 4. Separate features and labels
X = df["text"]
y = df["category"]

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "At least two categories with two or more examples each are required."
    )

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Build Linear SVM pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            random_state=42,
            max_iter=5000
        )
    )
])

# 7. Train model
model.fit(X_train, y_train)

# 8. Make predictions
predictions = model.predict(X_test)

# 9. Evaluate model
print("\nModel: Linear SVM")
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 3
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
labels = sorted(y.unique())

print(pd.DataFrame(
    confusion_matrix(y_test, predictions, labels=labels),
    index=labels,
    columns=labels
))

# 10. Show incorrect predictions
results = pd.DataFrame({
    "text": X_test.to_numpy(),
    "actual": y_test.to_numpy(),
    "predicted": predictions
})

errors = results[
    results["actual"] != results["predicted"]
]

print("\nMisclassified notices:")
if errors.empty:
    print("No misclassifications.")
else:
    print(errors.to_string(index=False))

# 11. Save trained model
joblib.dump(model, "notice_classifier.joblib")

print("\nModel saved as notice_classifier.joblib")
print("Supported categories:", list(model.classes_))
