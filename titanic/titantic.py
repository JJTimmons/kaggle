# %%
import pandas as pd

data = pd.read_csv("./titanic/train.csv")
data.columns = [x.lower() for x in data.columns]

data["has_cabin"] = data["cabin"].apply(lambda x: 0 if x != x else 1)
data["family_size"] = data["sibsp"] + data["parch"]
data["is_alone"] = data["family_size"].apply(lambda x: 1 if x == 0 else 0)
data["is_child"] = data["age"].apply(lambda x: 1 if x <= 16 else 0)

data.head()

# %%
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

floats = ["age", "fare", "sibsp", "parch", "family_size"]
float_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cats = ["pclass", "sex", "embarked", "has_cabin", "is_alone", "is_child"]
cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[("num", float_transformer, floats), ("cat", cat_transformer, cats)]
)

clf1 = Pipeline(
    [("preprocess", preprocessor), ("classify", XGBClassifier(n_estimators=100))]
)
clf2 = Pipeline(
    [("preprocess", preprocessor), ("classify", AdaBoostClassifier(n_estimators=100))]
)
clf3 = Pipeline(
    [
        ("preprocess", preprocessor),
        ("classify", RandomForestClassifier(n_estimators=100)),
    ]
)
eclf = VotingClassifier(estimators=[("xg", clf1), ("ad", clf2), ("rn", clf3)])

X = data.drop(["survived", "ticket", "passengerid", "name"], axis=1)
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

eclf.fit(X_train, y_train)

print("model score: %.3f\n" % eclf.score(X_test, y_test))


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

values = sorted(
    zip(X.columns, model.feature_importances_[: len(X.columns)]),
    key=lambda x: x[1],
    reverse=True,
)

x_pos = np.arange(len(values))

plt.bar(x_pos, [v[1] for v in values])
plt.xticks(x_pos, [v[0] for v in values], rotation=90)
plt.show()


# %%

test = pd.read_csv("./titanic/test.csv")
test.columns = [x.lower() for x in test.columns]

test_preds = eclf.predict(test)

with open("./titanic/prediction.csv", "w") as pred:
    pred.write("PassengerId,Survived\n")

    for i, p in test.iterrows():
        p_id = p["passengerid"]
        p_pred = test_preds[i]

        pred.write(f"{p_id},{p_pred}\n")
