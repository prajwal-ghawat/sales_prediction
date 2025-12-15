import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# ===================== DATA LOADING =====================

def load_data() -> pd.DataFrame:
    """Load CSV file safely from script directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "Advertising.csv")
    return pd.read_csv(data_path)


# ===================== EXPLORATORY ANALYSIS =====================

def explore_data(df: pd.DataFrame) -> None:

    print("\nDataset Preview:")
    print(df.head().to_string(index=False))

    print("\nDataset Info:")
    df.info()

    print("\nStatistical Summary:")
    print(df.describe())

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5
    )
    plt.title("Feature Correlation Matrix")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "feature_correlation.png"))
    plt.close()


# ===================== FEATURE / TARGET SPLIT =====================

def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ===================== MODEL TRAINING & EVALUATION =====================

def train_and_evaluate(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, color="darkgreen", alpha=0.7)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
        color="maroon",
        linewidth=2
    )

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.grid(True, linestyle=":", alpha=0.6)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "actual_vs_predicted_sales.png"))
    plt.close()

    return model


# ===================== MODEL SAVING =====================

def save_model(model):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "linear_regression_sales.pkl"))
    print("\nModel saved successfully.")


# ===================== MAIN =====================

def main():
    TARGET_COL = "Sales"

    df = load_data()
    explore_data(df)

    X, y = split_features_target(df, TARGET_COL)
    trained_model = train_and_evaluate(X, y)

    save_model(trained_model)


if __name__ == "__main__":
    main()

