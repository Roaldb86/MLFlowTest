import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.models.signature import infer_signature


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", default=0.5, type=float)
    parser.add_argument("l1_ratio", default=0.5, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    try:
        data = pd.read_csv(csv_url, sep=";")

    except Exception as e:
        print("Unable to read csv", e)

    train, test = train_test_split(data)

    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train.quality
    y_test = test.quality

    def eval_metrics(predictions, y_true):
        rmse_ = np.sqrt(mean_squared_error(y_true, predictions))
        mae_ = mean_absolute_error(y_true, predictions)
        r2_ = r2_score(y_true, predictions)
        return rmse_, mae_, r2_

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        predictions = lr.predict(X_test)
        rmse, mae, r2 = eval_metrics(predictions, y_test)

        print(f"ElasticNet model  alpha: {alpha}, l1_ratio: {l1_ratio}")
        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        signature = infer_signature(X_train, lr.predict(X_train))

        mlflow.sklearn.log_model(lr, "model", signature=signature)
