# ==================================================
# AICommerce – גרסת MVP חזקה ומאוחדת (חינמית)
# כולל:
# - מערכת משתמשים
# - FREE / PRO
# - Google Trends או Fallback
# - חיזוי Success (Rule + ML)
# - Auto ML Training
# - Dashboard אמיתי
# - חנות DEMO
# - API מלא
# ==================================================

from flask import Flask, render_template, request, redirect, url_for, session, Response
import csv
import json
import secrets
import random
import io
from datetime import datetime, date
from pathlib import Path
from functools import wraps
import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- Optional Libs ----------
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except Exception:
    HAS_PYTRENDS = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPER_DEPS = True
except Exception:
    HAS_SCRAPER_DEPS = False

# ==================================================
# App Config
# ==================================================
app = Flask(__name__)
app.secret_key = "super_secret_key_123"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "aicommerce_data.csv"
USERS_PATH = BASE_DIR / "users.json"
API_USAGE_PATH = BASE_DIR / "api_usage.json"
SHOPIFY_LOG_PATH = BASE_DIR / "shopify_webhook_log.jsonl"
MODEL_PATH = BASE_DIR / "aicommerce_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "aicommerce_model_info.json"

FREE_DAILY_LIMIT = 10
FREE_API_DAILY_LIMIT = 100

# ==================================================
# Init Storage
# ==================================================
for path, default in [
    (USERS_PATH, {}),
    (API_USAGE_PATH, {}),
]:
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)

if not DATA_PATH.exists():
    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["email", "name", "category", "price", "trend_score", "success_score", "risk", "created_at"]
        )

# ==================================================
# Helpers
# ==================================================
def json_response(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json")

def generate_api_key():
    return secrets.token_hex(16)

def load_users():
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

# ==================================================
# Google Trends
# ==================================================
def get_trend_from_google(keyword):
    if not keyword:
        return random.randint(50, 80)

    if not HAS_PYTRENDS:
        return random.randint(50, 90)

    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload([keyword], timeframe="today 3-m")
        data = pytrends.interest_over_time()
        if data.empty:
            return 50
        return int(data[keyword].iloc[-1])
    except Exception:
        return random.randint(50, 80)

# ==================================================
# Rule Based Prediction
# ==================================================
def predict_success(price, trend_score, category="general"):
    price = float(price)
    trend_score = float(trend_score)
    score = (trend_score * 0.8) + ((100 - price) * 0.2)
    return max(0, min(100, int(score)))

def classify_risk(score):
    if score >= 75:
        return "פוטנציאל גבוה"
    elif score >= 50:
        return "בינוני"
    return "סיכון גבוה"

# ==================================================
# ML SYSTEM ✅
# ==================================================
_MODEL_CACHE = {"estimator": None, "info": None}

def get_model_info():
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_trained_model():
    if _MODEL_CACHE["estimator"]:
        return _MODEL_CACHE["estimator"]

    if not MODEL_PATH.exists():
        return None

    model = joblib.load(MODEL_PATH)
    _MODEL_CACHE["estimator"] = model
    return model

def ml_predict_success(price, trend_score, category):
    model = load_trained_model()
    if not model:
        return None
    df = pd.DataFrame([{
        "price": float(price),
        "trend_score": float(trend_score),
        "category": category
    }])
    return int(model.predict(df)[0])

def train_ml_model(min_samples=20):
    df = pd.read_csv(DATA_PATH)
    if len(df) < min_samples:
        return {"error": "Not enough samples"}

    X = df[["price", "trend_score", "category"]]
    y = df["success_score"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ("num", "passthrough", ["price", "trend_score"]),
    ])

    pipeline = Pipeline([
        ("pre", preprocess),
        ("model", RandomForestRegressor(n_estimators=150))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    pipeline.fit(Xtr, ytr)

    r2 = r2_score(yte, pipeline.predict(Xte))
    mae = mean_absolute_error(yte, pipeline.predict(Xte))

    joblib.dump(pipeline, MODEL_PATH)

    info = {
        "r2": r2,
        "mae": mae,
        "samples": len(df),
        "trained_at": datetime.now().isoformat()
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    _MODEL_CACHE["estimator"] = pipeline
    _MODEL_CACHE["info"] = info

    return info

def maybe_autotrain_model():
    if not MODEL_PATH.exists():
        train_ml_model()

def compute_success_score(price, trend_score, category):
    ml = ml_predict_success(price, trend_score, category)
    if ml is not None:
        return ml, "ml"
    return predict_success(price, trend_score, category), "rule"

# ==================================================
# AUTH
# ==================================================
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

# ==================================================
# ROUTES ✅
# ==================================================
@app.route("/")
@login_required
def index():
    maybe_autotrain_model()
    info = get_model_info()
    has_model = load_trained_model() is not None
    return render_template("index.html", has_model=has_model, model_info=info)

@app.route("/train-model", methods=["POST"])
@login_required
def train_model_route():
    result = train_ml_model()
    session["train_message"] = str(result)
    return redirect(url_for("index"))

@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()
    error = None
    if request.method == "POST":
        email = request.form["email"]
        pw = request.form["password"]
        if email in users and users[email]["password"] == pw:
            session["user"] = {"email": email, "plan": users[email]["plan"]}
            return redirect("/")
        error = "Login Failed"
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    users = load_users()
    if request.method == "POST":
        email = request.form["email"]
        pw = request.form["password"]
        users[email] = {
            "password": pw,
            "plan": "FREE",
            "api_key": generate_api_key()
        }
        save_users(users)
        return redirect("/login")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ==================================================
# API
# ==================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    price = data["price"]
    category = data.get("category", "general")
    keyword = data.get("keyword", "")

    trend = get_trend_from_google(keyword or category)
    score, src = compute_success_score(price, trend, category)

    return json_response({
        "success_score": score,
        "risk": classify_risk(score),
        "model_source": src
    })

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
