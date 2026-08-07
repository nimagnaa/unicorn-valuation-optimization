from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT/"data/processed/forecast_dataset.csv"

MODEL = PROJECT_ROOT/"models/tuned/tuned_xgboost.pkl"

OUTPUT = PROJECT_ROOT/"outputs/optimization"

OUTPUT.mkdir(parents=True,exist_ok=True)

print("="*60)
print("AI OPTIMIZATION ENGINE")
print("="*60)

df=pd.read_csv(DATA)

model=joblib.load(MODEL)

results=[]

# ---------------------------------------------------

feature_columns=model.feature_names_in_

# ---------------------------------------------------

for _,row in df.iterrows():

    best_prediction=-1

    best=None

    for revenue in np.arange(1.00,1.31,0.05):

        for burn in np.arange(0.75,1.01,0.05):

            for hiring in np.arange(1.00,1.26,0.05):

                sample=row.copy()

                sample["Estimated_Revenue_Million"]*=revenue

                sample["Estimated_Burn_Million"]*=burn

                sample["Estimated_Employees"]*=hiring

                sample["Employee_Growth_Rate"]*=hiring

                sample=pd.DataFrame([sample])

                sample=sample.drop(

                    columns=[

                        "Company",
                        "Expected_Valuation_12M"

                    ],

                    errors="ignore"

                )

                sample=pd.get_dummies(sample)

                sample=sample.reindex(

                    columns=feature_columns,

                    fill_value=0

                )

                pred=model.predict(sample)[0]

                if pred>best_prediction:

                    best_prediction=pred

                    best={

                        "Revenue_Factor":revenue,

                        "Burn_Factor":burn,

                        "Hiring_Factor":hiring,

                        "Predicted_Valuation":pred

                    }

    results.append(best)

# ---------------------------------------------------

opt=pd.DataFrame(results)

final=pd.concat([df,opt],axis=1)

final["Recommended_Revenue"]=(
    final["Estimated_Revenue_Million"]
    *final["Revenue_Factor"]
)

final["Recommended_Burn"]=(
    final["Estimated_Burn_Million"]
    *final["Burn_Factor"]
)

final["Recommended_Employees"]=(
    final["Estimated_Employees"]
    *final["Hiring_Factor"]
)

SAVE=OUTPUT/"ml_optimization.csv"

final.to_csv(SAVE,index=False)

print()

print("Optimization Complete")

print()

print(SAVE)