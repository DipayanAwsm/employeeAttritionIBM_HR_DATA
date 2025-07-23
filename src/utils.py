import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from src.config import SELECTED_FEATURES_PATH

def preprocess_data(df, target_col):
    df = df.dropna()

    # Separate target and features
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Feature Selection: Select top 20 features based on mutual information
    selector = SelectKBest(mutual_info_classif, k=min(20, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_support = selector.get_support()
    selected_scores = selector.scores_[selected_support]
    selected_features = X.columns[selected_support]

    feature_df = pd.DataFrame({
        "SelectedFeatures": selected_features,
        "SelectionReason": [
            f"High mutual information score: {score:.4f}" for score in selected_scores
        ]
    })
    feature_df.to_csv(SELECTED_FEATURES_PATH, index=False)

    X = pd.DataFrame(X_selected, columns=selected_features)

    return train_test_split(X, y, test_size=0.2, random_state=42)
