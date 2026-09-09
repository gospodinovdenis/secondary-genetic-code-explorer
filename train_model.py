from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA = Path("secondary_genetic_code_dataset.csv")
df = pd.read_csv(DATA, keep_default_na=False)
df = df[df["length"] >= 59].copy()
df = df.drop_duplicates(subset=["sequence", "amino_acid"]).reset_index(drop=True)
pos_cols = [f"pos_{i}" for i in range(1, 86)]
X = df[pos_cols + ["length"]].copy()
for c in pos_cols:
    X[c] = X[c].replace("", "MISSING")
y = df["amino_acid"]
groups = df["sequence"]
pre = ColumnTransformer([
    ("seq", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), pos_cols),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["length"]),
])
pipe = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=700, class_weight="balanced", solver="lbfgs"))])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=groups))
pipe.fit(X.iloc[tr], y.iloc[tr])
pred = pipe.predict(X.iloc[te])
print("Grouped holdout accuracy:", accuracy_score(y.iloc[te], pred))
print("Grouped holdout macro-F1:", f1_score(y.iloc[te], pred, average="macro"))
print(classification_report(y.iloc[te], pred, zero_division=0))
pipe.fit(X, y)
joblib.dump(pipe, "sgc_model.joblib")
