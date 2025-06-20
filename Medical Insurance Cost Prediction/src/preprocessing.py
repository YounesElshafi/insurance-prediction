import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# 1. Load the dataset from a CSV file
# -----------------------------------------------
def load_data(path):
    """Load the dataset from a CSV file"""
    return pd.read_csv(path)


# -----------------------------------------------
# 2. Encode categorical features
# -----------------------------------------------
def encode_data(df):
    """Convert categorical variables to numerical representations"""
    df = df.copy()

    # Encode binary categorical features
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # One-hot encode 'region' column (drop one to avoid multicollinearity)
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    return df


# -----------------------------------------------
# 3. Split the dataset into features and target
# -----------------------------------------------
def split_features_targets(df, target='charges'):
    """Separate input features (X) and the target variable (y)"""
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


# -----------------------------------------------
# 4. Split the dataset into training and testing sets
# -----------------------------------------------
def split_train_test(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------------------------
# 5. Scale numerical features only
# -----------------------------------------------
def scale_numeric_features(X_train, X_test, numeric_cols):
    """Apply standard scaling to selected numerical columns"""
    scaler = StandardScaler()

    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Scale only the numeric columns
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler


# -----------------------------------------------
# 6. Split dataset based on smoker status
# -----------------------------------------------
def split_by_smoker(df):
    """Split the dataset into smoker and non-smoker groups"""
    smoker_df = df[df['smoker'] == 1].reset_index(drop=True)
    nonsmoker_df = df[df['smoker'] == 0].reset_index(drop=True)
    return smoker_df, nonsmoker_df


# -----------------------------------------------
# 7. Full preprocessing pipeline
# -----------------------------------------------
def preprocess_pipeline(path, target='charges'):
    """
    Complete data preprocessing pipeline:
    - Load data
    - Encode categorical variables
    - Split into features and target
    - Train/test split
    - Scale numeric features only
    """
    df = load_data(path)
    df = encode_data(df)
    X, y = split_features_targets(df, target=target)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    numeric_cols = ['age', 'bmi', 'children']
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test, numeric_cols)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
