# === File: main.py ===
from src.data_loader import load_data
from src.utils import preprocess_data
from src.model import train_and_evaluate_models
from src.config import TARGET_COLUMN
import pandas as pd

if __name__ == "__main__":
    df = load_data()
    print("\nData loaded successfully. Preview:")
    print(df.head())

    X_train, X_test, y_train, y_test = preprocess_data(df, TARGET_COLUMN)
    print("\nData preprocessing complete.")

    train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print("\nAll models trained, saved, and evaluated.")

# === End of Project ===
