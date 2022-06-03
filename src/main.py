""" Execution.
"""

import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


from visualize_residuals import visualize_residuals
from constants import (
    DATASET_NAME, 
    COLUMN_NAMES,
)

def main():
    # Load data
    df = pd.read_csv(
        os.path.join("./src/data", DATASET_NAME),
        names=COLUMN_NAMES,
        na_values="?",
        comment="\t",
        sep=" ",
        skipinitialspace=True,
    )

    df = df.drop(["origin"], axis="columns",).dropna()

    df_train = df.sample(frac=0.8, random_state=0)
    df_test = df.drop(df_train.index)

    x_train = df_train.drop(["mpg"], axis="columns",)
    y_train = df_train[["mpg"]]
    x_test = df_test.drop(["mpg"], axis="columns",)
    y_test = df_test[["mpg"]]

    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # df to catch predictions
    df_preds = pd.DataFrame()
    
    # First model
    model_lr = LinearRegression().fit(x_train_scaled, y_train)
    y_pred = model_lr.predict(x_test_scaled)
    df_preds["model_lr"] = y_pred.flatten()
    
    # Second model
    model_rf = RandomForestRegressor().fit(x_train_scaled, y_train)
    y_pred = model_rf.predict(x_test_scaled)
    df_preds["model_rf"] = y_pred.flatten()

    fig, _, _ = visualize_residuals(df_preds, y_test)
    fig.write_html("src/images/residual_distribution.html")
    fig.write_image("src/images/residual_distribution.png", scale=1,)


if __name__ == "__main__":
    main()
