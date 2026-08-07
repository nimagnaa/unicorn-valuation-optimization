from pathlib import Path

import joblib
import optuna

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

from src.tuning.data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "tuned"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT = PROJECT_ROOT / "outputs" / "tuning"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

X_train, X_test, y_train, y_test = load_data()


cv = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


def objective(trial):

    params = {

        "n_estimators":

            trial.suggest_int(
                "n_estimators",
                300,
                1500
            ),

        "learning_rate":

            trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True
            ),

        "max_depth":

            trial.suggest_int(
                "max_depth",
                3,
                12
            ),

        "min_child_weight":

            trial.suggest_int(
                "min_child_weight",
                1,
                10
            ),

        "subsample":

            trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

        "colsample_bytree":

            trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

        "gamma":

            trial.suggest_float(
                "gamma",
                0,
                5
            ),

        "reg_alpha":

            trial.suggest_float(
                "reg_alpha",
                0,
                2
            ),

        "reg_lambda":

            trial.suggest_float(
                "reg_lambda",
                0.5,
                5
            ),

        "random_state":42,

        "objective":"reg:squarederror"

    }

    model = XGBRegressor(**params)

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=cv,

        scoring="r2",

        n_jobs=-1

    ).mean()

    return score


print("="*60)
print("OPTUNA - XGBOOST")
print("="*60)

study = optuna.create_study(

    direction="maximize",

    sampler=optuna.samplers.TPESampler(seed=42),

    pruner=optuna.pruners.MedianPruner()

)

study.optimize(

    objective,

    n_trials=50,

    show_progress_bar=True

)

print()

print("Best R2")

print(study.best_value)

print()

print("Best Parameters")

print(study.best_params)

best_model = XGBRegressor(

    **study.best_params,

    random_state=42,

    objective="reg:squarederror"

)

best_model.fit(

    X_train,

    y_train

)

joblib.dump(

    best_model,

    MODEL_DIR/"tuned_xgboost.pkl"

)

pd.DataFrame(

    study.trials_dataframe()

).to_csv(

    OUTPUT/"xgboost_trials.csv",

    index=False

)

print()

print("Saved Model")

print(MODEL_DIR)

print()

print("Saved Trials")

print(OUTPUT)