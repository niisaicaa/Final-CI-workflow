import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import sys
import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    C_value = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    dataset_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data_banknote_preprocessing.csv"
        )
    )

    data = pd.read_csv(dataset_path)

    X = data.drop(columns="class")
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train.head(5)

    model = LogisticRegression(
        C=C_value,
        max_iter=max_iter,
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Manual logging (ADVANCED)
    mlflow.log_param("C", C_value)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    print(f"Accuracy: {accuracy:.2f}")