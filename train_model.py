# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# Load Dataset
path = Path(__file__).resolve().parent
dataset = pd.read_csv(path / "iris_dataset_100.csv")

# Rename Columns
dataset.columns = [
    'sepal_length', 'sepal_width',
    'petal_length', 'petal_width', 'species'
]

# Convert Species into Numbers
dataset['target'] = dataset['species'].str.lower().str.replace(
    'iris-', '', regex=False
).map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

if dataset['target'].isna().any():
    raise ValueError("Unknown species found in dataset")

# Feature Engineering
dataset['sepal_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_ratio'] = dataset['petal_length'] / dataset['petal_width']

# Features and Target
X = dataset.drop(['species', 'target'], axis=1)
y = dataset['target']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44, stratify=y
)

# Models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=44
    )
}

results = {}

# Train and Evaluate Models
for name, model in models.items():

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    predictions = model.predict(X_test)

    results[name] = {
        "Train Accuracy": accuracy_score(y_train, train_pred) * 100,
        "Test Accuracy": accuracy_score(y_test, predictions) * 100,
        "Precision": precision_score(y_test, predictions, average='weighted', zero_division=0) * 100,
        "Recall": recall_score(y_test, predictions, average='weighted', zero_division=0) * 100,
        "F1 Score": f1_score(y_test, predictions, average='weighted', zero_division=0) * 100
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=sorted(y.unique()),
        yticklabels=sorted(y.unique())
    )
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path / f'{name.replace(" ", "_")}_CM.png')
    plt.show()
    plt.close()

# Feature Importance
rf_model = models["Random Forest"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind='barh', figsize=(8, 5), color='teal')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(path / 'FeatureImportance.png')
plt.show()
plt.close()

# Save Scores
with open(path / 'scores.txt', 'w') as file:
    for name, scores in results.items():
        file.write(f'\n{name}\n')
        file.write('-' * 30 + '\n')
        for metric, value in scores.items():
            file.write(f'{metric}: {value:.2f}%\n')

# Final Results
print("\nMODEL PERFORMANCE")
print(pd.DataFrame(results).T.round(2))
print("\nProject Completed Successfully!")
print("Generated: Confusion Matrices, Feature Importance and scores.txt")