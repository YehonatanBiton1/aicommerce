import os
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

DATA_PATH = "aicommerce_data.csv"
MODEL_PATH = "aicommerce_model.pkl"
MODEL_INFO_PATH = "aicommerce_model_info.json"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["real_success"])

if len(df) < 20:
    print("❌ צריך לפחות 20 דוגמאות אמיתיות עם real_success")
    exit()

X = df[["price", "trend_score", "category", "orders_now"]]
y = df["real_success"].astype(int)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
    ("num", "passthrough", ["price", "trend_score", "orders_now"]),
])

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

joblib.dump(pipeline, MODEL_PATH)

info = {
    "trained_at": datetime.now().isoformat(),
    "n_samples": int(len(df)),
    "accuracy": float(acc),
    "auc": float(auc)
}

with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("✅ מודל אומן מדאטה אמיתי")
print(info)
