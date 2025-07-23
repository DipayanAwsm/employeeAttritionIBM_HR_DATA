# === File: src/model.py ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import os
from src.config import MODEL_FOLDER, REPORT_FOLDER, MODEL_NAMES

ALL_MODELS = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC()
}

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    for name in MODEL_NAMES:
        model = ALL_MODELS[name]
        model_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
        report_path = os.path.join(REPORT_FOLDER, f"{name}_report.csv")

        if os.path.exists(model_path):
            print(f"\nModel {name} already exists. Skipping training.")
            continue

        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(report_path)

        print(f"{name} model saved to {model_path}")
        print(f"{name} report saved to {report_path}")
