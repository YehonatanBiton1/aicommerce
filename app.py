# ==================================================
# AICommerce – גרסת MVP חזקה ומאוחדת (חינמית)
# ==================================================

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, Response
)
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

# ---------- Google Trends (אופציונלי) ----------
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except Exception:
    HAS_PYTRENDS = False

# ---------- Requests + BS4 לסקרייפינג (אופציונלי) ----------
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPER_DEPS = True
except Exception:
    HAS_SCRAPER_DEPS = False

# ==================================================
# הגדרות בסיס
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
# DEMO PRODUCTS
# ==================================================
DEMO_PRODUCTS = [
    {"id": 1, "name": "LED Galaxy Projector", "category": "חדר / אסתטיקה", "price": 119.0, "trend_score": 82,
     "image": "https://images.pexels.com/photos/2763921/pexels-photo-2763921.jpeg",
     "description": "מקרן כוכבים לחדר, מושלם ל-TikTok ולחדרים אסתטיים."},
    {"id": 2, "name": "Wireless Earbuds Pro", "category": "אודיו / גאדג'טים", "price": 89.0, "trend_score": 77,
     "image": "https://images.pexels.com/photos/7156888/pexels-photo-7156888.jpeg",
     "description": "אוזניות בלוטות' עם ביטול רעשים."},
    {"id": 3, "name": "Ring Light + Tripod", "category": "יוצרי תוכן", "price": 139.0, "trend_score": 88,
     "image": "https://images.pexels.com/photos/6898859/pexels-photo-6898859.jpeg",
     "description": "סט צילום ליוצרי תוכן."},
    {"id": 4, "name": "Resistance Bands Set", "category": "כושר ביתי", "price": 79.0, "trend_score": 73,
     "image": "https://images.pexels.com/photos/6456319/pexels-photo-6456319.jpeg",
     "description": "סט גומיות לאימון ביתי."},
    {"id": 5, "name": "Car Phone Holder 360°", "category": "אביזרי רכב", "price": 59.0, "trend_score": 69,
     "image": "https://images.pexels.com/photos/799443/pexels-photo-799443.jpeg",
     "description": "מחזיק טלפון לרכב."},
    {"id": 6, "name": "Minimalist Desk Lamp", "category": "עיצוב שולחן", "price": 99.0, "trend_score": 75,
     "image": "https://images.pexels.com/photos/4475921/pexels-photo-4475921.jpeg",
     "description": "מנורת שולחן מודרנית."},
]

# ==================================================
# JSON Response Helper
# ==================================================
def json_response(data, status=200):
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json; charset=utf-8",
    )

# ==================================================
# SAFE FILE INIT (FULL FIX)
# ==================================================
if not USERS_PATH.exists():
    USERS_PATH.write_text("{}", encoding="utf-8")

if not API_USAGE_PATH.exists():
    API_USAGE_PATH.write_text("{}", encoding="utf-8")

REQUIRED_COLUMNS = ["email", "name", "category", "price", "trend_score",
                    "success_score", "risk", "created_at"]

if not DATA_PATH.exists():
    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(REQUIRED_COLUMNS)
else:
    try:
        df_fix = pd.read_csv(DATA_PATH)
        for col in REQUIRED_COLUMNS:
            if col not in df_fix.columns:
                df_fix[col] = ""
        df_fix = df_fix[REQUIRED_COLUMNS]
        df_fix.to_csv(DATA_PATH, index=False)
    except Exception:
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(REQUIRED_COLUMNS)

# ==================================================
# USER MANAGEMENT
# ==================================================
def load_users():
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))

def save_users(users):
    USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

def generate_api_key():
    return secrets.token_hex(16)

# ==================================================
# API USAGE
# ==================================================
def load_api_usage():
    return json.loads(API_USAGE_PATH.read_text(encoding="utf-8"))

def save_api_usage(data):
    API_USAGE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def check_api_limit(api_user):
    usage = load_api_usage()
    key = api_user["api_key"]
    today = str(date.today())

    entry = usage.get(key, {"date": today, "count": 0})
    if entry["date"] != today:
        entry = {"date": today, "count": 0}

    if api_user.get("plan") == "FREE" and entry["count"] >= FREE_API_DAILY_LIMIT:
        return False

    entry["count"] += 1
    usage[key] = entry
    save_api_usage(usage)
    return True

# ==================================================
# DECORATORS
# ==================================================
def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper

def api_key_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        users = load_users()
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")

        if request.is_json and not api_key:
            api_key = (request.get_json() or {}).get("api_key")

        if not api_key:
            return json_response({"error": "Missing API key"}, 401)

        api_user = None
        for email, u in users.items():
            if u.get("api_key") == api_key:
                api_user = {"email": email, "plan": u.get("plan", "FREE"), "api_key": api_key}
                break

        if not api_user or not check_api_limit(api_user):
            return json_response({"error": "API limit / invalid key"}, 401)

        kw["api_user"] = api_user
        return fn(*a, **kw)
    return wrapper

# ==================================================
# AI CORE
# ==================================================
def get_trend_from_google(keyword):
    if not keyword or not HAS_PYTRENDS:
        return random.randint(50, 90)
    try:
        pt = TrendReq()
        pt.build_payload([keyword], timeframe="today 3-m")
        data = pt.interest_over_time()
        return float(data[keyword].iloc[-1]) if not data.empty else 50
    except Exception:
        return random.randint(50, 80)

def predict_success(price, trend, category="general"):
    return max(0, min(100, int((trend * 0.8) + ((100 - price) * 0.2))))

def classify_risk(score):
    if score >= 75:
        return "פוטנציאל גבוה"
    if score >= 50:
        return "בינוני"
    return "סיכון גבוה"

# ==================================================
# ML SAFE LAYER (FINAL SAFE)
# ==================================================
_MODEL_CACHE = {"estimator": None}

def get_model_info():
    if not MODEL_INFO_PATH.exists():
        return None
    try:
        return json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_trained_model():
    if _MODEL_CACHE["estimator"] is not None:
        return _MODEL_CACHE["estimator"]

    if not MODEL_PATH.exists():
        return None

    try:
        model = joblib.load(MODEL_PATH)
        _MODEL_CACHE["estimator"] = model
        return model
    except Exception:
        try:
            MODEL_PATH.unlink(missing_ok=True)
            MODEL_INFO_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        return None

def train_ml_model(min_samples=20):
    df = pd.read_csv(DATA_PATH)
    if len(df) < min_samples:
        return {"error": "Not enough data"}

    X = df[["price", "trend_score", "category"]]
    y = df["success_score"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ("num", "passthrough", ["price", "trend_score"])
    ])

    model = RandomForestRegressor(n_estimators=200)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    pipe.fit(X_tr, y_tr)

    joblib.dump(pipe, MODEL_PATH)

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(df)),
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _MODEL_CACHE["estimator"] = pipe

    return {"info": info}

def maybe_autotrain_model():
    if not DATA_PATH.exists():
        return
    df = pd.read_csv(DATA_PATH)
    if len(df) >= 20 and not MODEL_PATH.exists():
        train_ml_model()

def compute_success_score(price, trend, category):
    model = load_trained_model()
    if model:
        try:
            pred = model.predict(pd.DataFrame([{
                "price": price,
                "trend_score": trend,
                "category": category
            }]))
            return int(pred[0]), "ml"
        except Exception:
            pass

    return predict_success(price, trend, category), "rule"

# ==================================================
# ROUTES — AUTH + UI + API
# ==================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()
    error = None
    if request.method == "POST":
        email = request.form["email"].lower()
        pw = request.form["password"]
        if email in users and users[email]["password"] == pw:
            session["user"] = {"email": email, "plan": users[email].get("plan", "FREE")}
            return redirect(url_for("index"))
        error = "פרטי התחברות שגויים"
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    users = load_users()
    error = None
    if request.method == "POST":
        email = request.form["email"].lower()
        pw = request.form["password"]
        if email in users:
            error = "משתמש קיים"
        else:
            users[email] = {"password": pw, "plan": "FREE", "api_key": generate_api_key()}
            save_users(users)
            return redirect(url_for("login"))
    return render_template("register.html", error=error)

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    users = load_users()
    msg = None
    if request.method == "POST":
        email = request.form["email"].lower()
        new_pw = request.form["new_password"]
        if email in users:
            users[email]["password"] = new_pw
            save_users(users)
            msg = "עודכן בהצלחה"
    return render_template("forgot.html", success=msg)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    email = session["user"]["email"]
    plan = session["user"]["plan"]

    used_today = get_user_daily_usage(email)
    limit_today = FREE_DAILY_LIMIT if plan == "FREE" else None

    maybe_autotrain_model()
    model_info = get_model_info() or {}
    has_model = load_trained_model() is not None

    history = load_predictions(email, limit=20)
    dashboard = build_dashboard()
    user_insights = build_user_insights(email)
    free_ideas = generate_free_ideas()

    result = session.pop("last_result", None)
    compare_result = session.pop("compare_result", None)
    train_message = session.pop("train_message", None)

    return render_template(
        "index.html",
        user=session["user"],
        used_today=used_today,
        limit_today=limit_today,
        history=history,
        dashboard=dashboard,
        user_insights=user_insights,
        free_ideas=free_ideas,
        result=result,
        compare_result=compare_result,
        has_model=has_model,
        model_info=model_info,
        train_message=train_message,
    )

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
