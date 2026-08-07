from pathlib import Path

import joblib
import optuna
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from catboost import CatBoostRegressor

from src.tuning.data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "tuned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = PROJECT_ROOT / "outputs" / "tuning"
OUTPUT.mkdir(parents=True, exist_ok=True)

X_train, X_test, y_train, y_test = load_data()

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def objective(trial):

    params = {

        "iterations":
            trial.suggest_int("iterations",300,1500),

        "depth":
            trial.suggest_int("depth",4,10),

        "learning_rate":
            trial.suggest_float("learning_rate",0.01,0.20,log=True),

        "l2_leaf_reg":
            trial.suggest_float("l2_leaf_reg",1,10),

        "random_strength":
            trial.suggest_float("random_strength",0,5),

        "bagging_temperature":
            trial.suggest_float("bagging_temperature",0,5),

        "loss_function":"RMSE",

        "verbose":False,

        "random_seed":42

    }

    model = CatBoostRegressor(**params)

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
print("OPTUNA - CATBOOST")
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
print("Best CV R2 :", study.best_value)

print()
print("Best Parameters")
print(study.best_params)

best_model = CatBoostRegressor(
    **study.best_params,
    loss_function="RMSE",
    verbose=False,
    random_seed=42
)

best_model.fit(
    X_train,
    y_train
)

joblib.dump(
    best_model,
    MODEL_DIR/"tuned_catboost.pkl"
)

pd.DataFrame(
    study.trials_dataframe()
).to_csv(
    OUTPUT/"catboost_trials.csv",
    index=False
)

print()
print("Model Saved")
print(MODEL_DIR)

print()
print("Trials Saved")
print(OUTPUT)