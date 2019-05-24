# %%
import pandas as pd

pd.set_option("display.max_columns", None)

data = pd.read_csv("./houses/train.csv")
data.columns = [c.lower() for c in data.columns]
data.sample(25)

# %%
from sklearn.model_selection import train_test_split
import numpy as np

np.set_printoptions(suppress=True)

ranges = []

nums = [
    "lotfrontage",
    "lotarea",
    "masvnrarea",
    "1stflrsf",
    "2ndflrsf",
    "miscval",
    "bsmtunfsf",
    "yearbuilt",
    "yearremodadd",
    "mosold",
    "yrsold",
]

cats = [
    "overallqual",
    "overallcond",
    "garagecars",
    "mssubclass",
    "mszoning",
    "street",
    "alley",
    "lotshape",
    "landcontour",
    "utilities",
    "landslope",
    "neighborhood",
    "bldgtype",
    "housestyle",
    "roofstyle",
    "exterior1st",
    "exterior2nd",
    "exterqual",
    "foundation",
    "saletype",
    "salecondition",
    "paveddrive",
    "centralair",
    "garagetype",
    "bsmtfullbath",
    "bsmthalfbath",
    "functional",
]

X = data[ranges + nums + cats]
y = data["saleprice"]
np.log1p(y)


# %%
import random
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

range_transform = Pipeline(
    [("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())]
)

num_transform = Pipeline(
    [
        ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scale", StandardScaler()),
    ]
)

cat_transform = Pipeline(
    [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    [
        ("ranges", range_transform, ranges),
        ("cats", cat_transform, cats),
        ("nums", num_transform, nums),
    ]
)

reg = Pipeline(
    [
        ("preprocess", preprocess),
        ("regress", XGBRegressor(booster="gbtree", objective="reg:squarederror")),
    ]
)

grid = GridSearchCV(reg, {}, cv=10, n_jobs=-1)
grid.fit(X, y)

print(f"accuracy: {grid.best_score_}")

model = grid.best_estimator_.steps[1][1]
for label, impact in zip(X.columns, model.feature_importances_):
    print(label, impact)


# %%
from sklearn.model_selection import GridSearchCV, cross_val_predict

test = pd.read_csv("./houses/test.csv")
test.columns = [c.lower() for c in test.columns]

preds = grid.predict(test)
np.expm1(preds)

with open("./houses/prediction.csv", "w") as pred:
    pred.write("Id,SalePrice\n")

    for i, p in test.iterrows():
        p_id = p["id"]
        p_pred = preds[i]

        pred.write(f"{p_id},{p_pred}\n")
