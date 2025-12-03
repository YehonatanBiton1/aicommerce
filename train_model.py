# train_model.py
import os
import json
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

from data_cleaning import clean_training_frame

DATA_PATH = "aicommerce_data.csv"
MODEL_PATH = "aicommerce_model.pkl"
MODEL_INFO_PATH = "aicommerce_model_info.json"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ לא נמצא קובץ נתונים: {DATA_PATH}")
        return

    raw_df = pd.read_csv(DATA_PATH)

    required_cols = ["price", "trend_score", "category", "success_score"]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        print("❌ חסרות עמודות:", missing)
        return

    df = clean_training_frame(raw_df)

    if len(df) < 15:
        print(
            "❌ צריך לפחות 15 מוצרים נקיים כדי לאמן מודל טוב (כרגע יש",
            len(df),
            "מ-",
            len(raw_df),
            "שורות)",
        )
        return

    X = df[["price", "trend_score", "category"]]
    y = df["success_score"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("num", "passthrough", ["price", "trend_score"]),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("מאמן מודל על הדאטה...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("מריץ Cross Validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    print("\n=== תוצאות אימון ===")
    print(f"R² על סט טסט: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Cross-Val ממוצע: {cv_mean:.3f} (סטיית תקן {cv_std:.3f})")

    joblib.dump(pipeline, MODEL_PATH)

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(df)),
        "r2_test": float(r2),
        "mae_test": float(mae),
        "r2_cv_mean": cv_mean,
        "r2_cv_std": cv_std,
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ המודל נשמר אל: {MODEL_PATH}")
    print(f"ℹ️ מידע על המודל נשמר אל: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    main()
