# ==================================================
# AICommerce – FULL MVP SAFE FINAL VERSION
# ==================================================

from flask import Flask, render_template, request, redirect, url_for, session, Response
import csv, json, secrets, random, io, os
from datetime import datetime, date
from pathlib import Path
from functools import wraps

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# OPTIONAL LIBS
# ==================================================
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except:
    HAS_PYTRENDS = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPER = True
except:
    HAS_SCRAPER = False

# ==================================================
# BASIC CONFIG
# ==================================================

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "aicommerce_data.csv"
USERS_PATH = BASE_DIR / "users.json"
API_USAGE_PATH = BASE_DIR / "api_usage.json"
MODEL_PATH = BASE_DIR / "aicommerce_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "aicommerce_model_info.json"

FREE_DAILY_LIMIT = 10
FREE_API_DAILY_LIMIT = 100

# ==================================================
# SAFE FILE INIT (CRITICAL FIX)
# ==================================================

if not USERS_PATH.exists():
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)

if not API_USAGE_PATH.exists():
    with open(API_USAGE_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)

REQUIRED_COLUMNS = [
    "email",
    "name",
    "category",
    "price",
    "trend_score",
    "success_score",
    "risk",
    "created_at",
]

if not DATA_PATH.exists():
    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(REQUIRED_COLUMNS)
else:
    # ✅ תיקון אוטומטי אם חסרות עמודות
    try:
        df_check = pd.read_csv(DATA_PATH)
        for col in REQUIRED_COLUMNS:
            if col not in df_check.columns:
                df_check[col] = ""
        df_check = df_check[REQUIRED_COLUMNS]
        df_check.to_csv(DATA_PATH, index=False)
    except:
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(REQUIRED_COLUMNS)

# ==================================================
# HELPERS
# ==================================================

def json_response(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json")

def load_users():
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def generate_api_key():
    return secrets.token_hex(16)

# ==================================================
# API LIMIT
# ==================================================

def load_api_usage():
    with open(API_USAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_api_usage(data):
    with open(API_USAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_api_limit(api_user):
    usage = load_api_usage()
    key = api_user["api_key"]
    today = str(date.today())

    entry = usage.get(key, {"date": today, "count": 0})
    if entry["date"] != today:
        entry = {"date": today, "count": 0}

    if api_user["plan"] == "FREE" and entry["count"] >= FREE_API_DAILY_LIMIT:
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
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        users = load_users()

        user = None
        for email, u in users.items():
            if u.get("api_key") == api_key:
                user = {"email": email, "plan": u.get("plan", "FREE"), "api_key": api_key}

        if not user or not check_api_limit(user):
            return json_response({"error": "API limit or invalid key"}, 401)

        kw["api_user"] = user
        return fn(*a, **kw)
    return wrapper

# ==================================================
# AI ENGINE
# ==================================================

def get_trend_from_google(keyword):
    if not keyword:
        return random.randint(40, 80)
    if not HAS_PYTRENDS:
        return random.randint(50, 90)
    try:
        pytrends = TrendReq()
        pytrends.build_payload([keyword], timeframe="today 3-m")
        val = pytrends.interest_over_time()
        return float(val[keyword].iloc[-1]) if not val.empty else 50
    except:
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
# ML SAFE LAYER (CRASH PROOF)
# ==================================================

_MODEL_CACHE = {"estimator": None}

def load_trained_model():
    if _MODEL_CACHE["estimator"] is not None:
        return _MODEL_CACHE["estimator"]

    if not MODEL_PATH.exists():
        return None

    try:
        model = joblib.load(MODEL_PATH)
        _MODEL_CACHE["estimator"] = model
        return model
    except:
        MODEL_PATH.unlink(missing_ok=True)
        MODEL_INFO_PATH.unlink(missing_ok=True)
        return None

def train_ml_model():
    df = pd.read_csv(DATA_PATH)
    if len(df) < 20:
        return

    X = df[["price", "trend_score", "category"]]
    y = df["success_score"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ("num", "passthrough", ["price", "trend_score"])
    ])

    model = RandomForestRegressor(n_estimators=200)
    pipe = Pipeline([("pre", preprocess), ("model", model)])

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2)
    pipe.fit(X_tr, y_tr)

    joblib.dump(pipe, MODEL_PATH)
    _MODEL_CACHE["estimator"] = pipe

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
        except:
            pass
    return predict_success(price, trend, category), "rule"

# ==================================================
# AUTH
# ==================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()
    if request.method == "POST":
        email = request.form["email"].lower()
        pw = request.form["password"]

        if email in users and users[email]["password"] == pw:
            session["user"] = {"email": email, "plan": users[email]["plan"]}
            return redirect(url_for("index"))

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    users = load_users()
    if request.method == "POST":
        email = request.form["email"].lower()
        pw = request.form["password"]

        users[email] = {"password": pw, "plan": "FREE", "api_key": generate_api_key()}
        save_users(users)
        return redirect(url_for("login"))

    return render_template("register.html")

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
            msg = "הסיסמה עודכנה בהצלחה"

    return render_template("forgot.html", success=msg)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==================================================
# UI
# ==================================================

@app.route("/")
@login_required
def index():
    df = pd.read_csv(DATA_PATH)

    if "email" in df.columns:
        history = df[df["email"] == session["user"]["email"]].tail(20).to_dict("records")
    else:
        history = []

    return render_template("index.html", history=history, user=session["user"])

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    email = session["user"]["email"]
    name = request.form["name"]
    category = request.form["category"]
    price = float(request.form["price"])
    trend = get_trend_from_google(name)

    score, src = compute_success_score(price, trend, category)
    risk = classify_risk(score)

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            email,
            name,
            category,
            price,
            trend,
            score,
            risk,
            datetime.now().isoformat()
        ])

    train_ml_model()
    return redirect(url_for("index"))

# ==================================================
# API
# ==================================================

@app.route("/api/predict", methods=["POST"])
@api_key_required
def api_predict(api_user):
    data = request.get_json(force=True)

    price = float(data.get("price", 0))
    category = data.get("category", "general")
    keyword = data.get("keyword", "")

    trend = get_trend_from_google(keyword)
    score, src = compute_success_score(price, trend, category)
    risk = classify_risk(score)

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            api_user["email"],
            keyword,
            category,
            price,
            trend,
            score,
            risk,
            datetime.now().isoformat()
        ])

    train_ml_model()

    return json_response({
        "success_score": score,
        "risk": risk,
        "model_source": src
    })

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
