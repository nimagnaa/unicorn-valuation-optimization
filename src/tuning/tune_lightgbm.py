from pathlib import Path

import joblib
import optuna
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor

from src.tuning.data_loader import load_data

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "tuned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = PROJECT_ROOT / "outputs" / "tuning"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

X_train, X_test, y_train, y_test = load_data()

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================================
# OBJECTIVE
# ==========================================================

def objective(trial):

    params = {

        "n_estimators":
            trial.suggest_int("n_estimators",300,1500),

        "learning_rate":
            trial.suggest_float("learning_rate",0.01,0.20,log=True),

        "num_leaves":
            trial.suggest_int("num_leaves",20,200),

        "max_depth":
            trial.suggest_int("max_depth",3,15),

        "min_child_samples":
            trial.suggest_int("min_child_samples",5,50),

        "subsample":
            trial.suggest_float("subsample",0.6,1.0),

        "colsample_bytree":
            trial.suggest_float("colsample_bytree",0.6,1.0),

        "reg_alpha":
            trial.suggest_float("reg_alpha",0.0,2.0),

        "reg_lambda":
            trial.suggest_float("reg_lambda",0.0,5.0),

        "random_state":42,

        "verbosity":-1

    }

    model = LGBMRegressor(**params)

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
print("OPTUNA - LIGHTGBM")
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

best_model = LGBMRegressor(
    **study.best_params,
    random_state=42,
    verbosity=-1
)

best_model.fit(
    X_train,
    y_train
)

joblib.dump(
    best_model,
    MODEL_DIR/"tuned_lightgbm.pkl"
)

pd.DataFrame(
    study.trials_dataframe()
).to_csv(
    OUTPUT/"lightgbm_trials.csv",
    index=False
)

print()
print("Model Saved")
print(MODEL_DIR)

print()
print("Trials Saved")
print(OUTPUT)