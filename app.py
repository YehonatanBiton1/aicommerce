# ==================================================
# AICommerce – גרסת MVP חזקה ומאוחדת (חינמית)
# כולל:
# - מערכת משתמשים (Login / Register / Forgot)
# - FREE / PRO + הגבלת שימוש
# - Google Trends (אם מותקן) או Fallback אוטומטי
# - חיזוי Success למוצר (UI + API)
# - השוואת שני מוצרים
# - Dashboard מדאטה אמיתי (CSV)
# - חנות DEMO עם תמונות
# - Amazon Scraper (אם אפשר) + Fallback Fake
# - AliExpress Fake עם תמונות
# - TikTok Trends Fake
# - Winning Product
# - Top Products
# - Live Dashboard JSON
# - API Docs
# - Shopify Webhook (רק לוג)
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

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from data_cleaning import (
    DEFAULT_MIN_TRAINING_SAMPLES,
    clean_training_frame,
    validate_training_data,
)

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
app.secret_key = "super_secret_key_123"  # תחליף בקוד אמיתי

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "aicommerce_data.csv"
USERS_PATH = BASE_DIR / "users.json"
API_USAGE_PATH = BASE_DIR / "api_usage.json"
SHOPIFY_LOG_PATH = BASE_DIR / "shopify_webhook_log.jsonl"
MODEL_PATH = BASE_DIR / "aicommerce_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "aicommerce_model_info.json"

FREE_DAILY_LIMIT = 10          # שימושים ב-UI ליום
FREE_API_DAILY_LIMIT = 100     # קריאות API ליום

# ==================================================
# מוצרי DEMO לחנות (עם תמונות אמיתיות חינמיות)
# ==================================================
DEMO_PRODUCTS = [
    {
        "id": 1,
        "name": "LED Galaxy Projector",
        "category": "חדר / אסתטיקה",
        "price": 119.0,
        "trend_score": 82,
        "image": "https://images.pexels.com/photos/2763921/pexels-photo-2763921.jpeg",
        "description": "מקרן כוכבים לחדר, מושלם ל-TikTok ולחדרים אסתטיים."
    },
    {
        "id": 2,
        "name": "Wireless Earbuds Pro",
        "category": "אודיו / גאדג'טים",
        "price": 89.0,
        "trend_score": 77,
        "image": "https://images.pexels.com/photos/7156888/pexels-photo-7156888.jpeg",
        "description": "אוזניות בלוטות' עם ביטול רעשים, מתאימות ליום-יום."
    },
    {
        "id": 3,
        "name": "Ring Light + Tripod",
        "category": "יוצרי תוכן",
        "price": 139.0,
        "trend_score": 88,
        "image": "https://images.pexels.com/photos/6898859/pexels-photo-6898859.jpeg",
        "description": "סט רינג לייט + חצובה לצילום רילס, TikTok ולייבים."
    },
    {
        "id": 4,
        "name": "Resistance Bands Set",
        "category": "כושר ביתי",
        "price": 79.0,
        "trend_score": 73,
        "image": "https://images.pexels.com/photos/6456319/pexels-photo-6456319.jpeg",
        "description": "סט גומיות התנגדות לאימון מלא בבית – טרנד חזק."
    },
    {
        "id": 5,
        "name": "Car Phone Holder 360°",
        "category": "אביזרי רכב",
        "price": 59.0,
        "trend_score": 69,
        "image": "https://images.pexels.com/photos/799443/pexels-photo-799443.jpeg",
        "description": "מחזיק טלפון מסתובב לרכב, מתאים לכל הסמארטפונים."
    },
    {
        "id": 6,
        "name": "Minimalist Desk Lamp",
        "category": "עיצוב שולחן",
        "price": 99.0,
        "trend_score": 75,
        "image": "https://images.pexels.com/photos/4475921/pexels-photo-4475921.jpeg",
        "description": "מנורת שולחן מודרנית, מושלמת לחדר עבודה / לימודים."
    },
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
# יצירת קבצים ריקים אם חסרים
# ==================================================
if not USERS_PATH.exists():
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

if not API_USAGE_PATH.exists():
    with open(API_USAGE_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

if not DATA_PATH.exists():
    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "email",
                "name",
                "category",
                "price",
                "trend_score",
                "success_score",
                "risk",
                "created_at",
            ]
        )

# ==================================================
# User Management
# ==================================================
def load_users():
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def generate_api_key():
    return secrets.token_hex(16)

# ==================================================
# API Usage Limits
# ==================================================
def load_api_usage():
    with open(API_USAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_api_usage(usage):
    with open(API_USAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)


def check_api_limit(api_user):
    usage = load_api_usage()
    key = api_user["api_key"]
    today = str(date.today())

    entry = usage.get(key, {"date": today, "count": 0})
    if entry["date"] != today:
        entry = {"date": today, "count": 0}

    # PRO – בלי מגבלה
    if api_user.get("plan", "FREE") != "FREE":
        entry["count"] += 1
        usage[key] = entry
        save_api_usage(usage)
        return True

    # FREE – מגבלה יומית
    if entry["count"] >= FREE_API_DAILY_LIMIT:
        return False

    entry["count"] += 1
    usage[key] = entry
    save_api_usage(usage)
    return True

# ==================================================
# Decorators
# ==================================================
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


def api_key_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        users = load_users()

        api_key = (
            request.headers.get("X-API-Key")
            or request.args.get("api_key")
        )

        if api_key is None and request.is_json:
            body = request.get_json(silent=True) or {}
            api_key = body.get("api_key")

        if not api_key:
            return json_response({"error": "Missing API key"}, 401)

        api_user = None
        for email, data in users.items():
            if data.get("api_key") == api_key:
                api_user = {
                    "email": email,
                    "plan": data.get("plan", "FREE"),
                    "api_key": api_key,
                }
                break

        if api_user is None:
            return json_response({"error": "Invalid API key"}, 401)

        if not check_api_limit(api_user):
            return json_response({"error": "Daily API limit reached (FREE)"}, 429)

        kwargs["api_user"] = api_user
        return fn(*args, **kwargs)

    return wrapper

# ==================================================
# AI Engine – Google Trends + Success Score
# ==================================================
def get_trend_from_google(keyword: str) -> float:
    keyword = (keyword or "").strip()
    if not keyword:
        return random.randint(40, 80)

    if not HAS_PYTRENDS:
        # אין pytrends מותקן – נחזיר מספר רנדומלי "חכם"
        return random.randint(50, 90)

    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload([keyword], timeframe="today 3-m", geo="")
        data = pytrends.interest_over_time()
        if data.empty:
            return 50.0
        value = float(data[keyword].iloc[-1])
        return max(0.0, min(100.0, value))
    except Exception:
        return random.randint(50, 80)


def predict_success(price: float, trend_score: float, category: str = "general") -> int:
    """חישוב פשוט: 80% טרנד, 20% מחיר (זול יותר = טוב)."""
    try:
        price = float(price)
    except Exception:
        price = 0.0

    try:
        trend_score = float(trend_score)
    except Exception:
        trend_score = 50.0

    score = (trend_score * 0.8) + (max(0.0, 100.0 - price) * 0.2)
    score = int(round(score))
    return max(0, min(100, score))


def classify_risk(score: int) -> str:
    if score >= 75:
        return "פוטנציאל גבוה"
    elif score >= 50:
        return "בינוני"
    return "סיכון גבוה"


# ==================================================
# ML Model Training + Inference (חינמי)
# ==================================================
_MODEL_CACHE: dict[str, object | None] = {"estimator": None, "info": None}


def get_model_info():
    if not MODEL_INFO_PATH.exists():
        return None
    try:
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
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
        return None


def _clamp_score(val: float) -> int:
    try:
        score = int(round(val))
    except Exception:
        score = 0
    return max(0, min(100, score))


def ml_predict_success(price: float, trend_score: float, category: str):
    model = load_trained_model()
    if model is None:
        return None
    try:
        pred = model.predict(
            pd.DataFrame(
                [
                    {
                        "price": float(price),
                        "trend_score": float(trend_score),
                        "category": category or "general",
                    }
                ]
            )
        )
        return _clamp_score(float(pred[0]))
    except Exception:
        return None
def train_ml_model(min_samples: int = DEFAULT_MIN_TRAINING_SAMPLES):
    if not DATA_PATH.exists():
        return {"error": "אין דאטה לאימון"}

    df, error = validate_training_data(pd.read_csv(DATA_PATH), min_samples=min_samples)
    if error:
        return {"error": error}

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
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    joblib.dump(pipeline, MODEL_PATH)

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(df)),
        "r2_test": float(r2),
        "mae_test": float(mae),
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    _MODEL_CACHE["estimator"] = pipeline
    _MODEL_CACHE["info"] = info
    return {"info": info}


def maybe_autotrain_model():
    if not DATA_PATH.exists():
        return None

    df, error = validate_training_data(
        pd.read_csv(DATA_PATH), min_samples=DEFAULT_MIN_TRAINING_SAMPLES
    )
    if error:
        return None

    info = get_model_info() or {}
    already_samples = int(info.get("n_samples", 0))

    if already_samples >= len(df):
        return None

    return train_ml_model(min_samples=DEFAULT_MIN_TRAINING_SAMPLES)


def compute_success_score(price: float, trend_score: float, category: str):
    ml_score = ml_predict_success(price, trend_score, category)
    if ml_score is not None:
        return ml_score, "ml"
    return predict_success(price, trend_score, category), "rule"


def generate_free_ideas(n: int = 5):
    """יוצר רעיונות מוצרים חינמיים מהדגמות קיימות + קומבינציות."""
    ideas = []
    base = enrich_demo_products(DEMO_PRODUCTS)
    random.shuffle(base)

    extras = [
        {
            "name": "Content Creator Kit",
            "category": "יוצרי תוכן",
            "price": 219,
            "trend_score": 85,
        },
        {
            "name": "Eco Friendly Kitchen Set",
            "category": "בית",
            "price": 129,
            "trend_score": 78,
        },
        {
            "name": "Car Organizer 2-in-1",
            "category": "אביזרי רכב",
            "price": 69,
            "trend_score": 71,
        },
    ]

    for item in extras:
        item["success_score"] = predict_success(
            item["price"], item["trend_score"], item["category"]
        )
        item["risk"] = classify_risk(item["success_score"])

    combined = base + extras
    combined = sorted(
        combined,
        key=lambda p: (p["success_score"], p.get("trend_score", 0)),
        reverse=True,
    )

    for prod in combined[:n]:
        ideas.append(
            {
                "name": prod.get("name") or prod.get("title"),
                "category": prod.get("category", "general"),
                "price": prod.get("price", 0),
                "trend_score": prod.get("trend_score", 50),
                "success_score": prod.get("success_score", 0),
                "risk": prod.get("risk", ""),
                "description": prod.get(
                    "description",
                    "מוצר בעל פוטנציאל גבוה שזמין כעת כחלק מהחבילה החינמית.",
                ),
                "image": prod.get("image"),
            }
        )

    return ideas

# ==================================================
# שימוש יומי ב-UI למנוי FREE
# ==================================================
def get_user_daily_usage(email: str) -> int:
    if not DATA_PATH.exists():
        return 0

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return 0

    if "email" not in df.columns or "created_at" not in df.columns:
        return 0

    user_df = df[df["email"] == email]
    if user_df.empty:
        return 0

    dates = pd.to_datetime(user_df["created_at"], errors="coerce").dt.date
    today = date.today()
    return int((dates == today).sum())


def load_predictions(email: str | None = None, limit: int = 50):
    if not DATA_PATH.exists():
        return []

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return []

    if email and "email" in df.columns:
        df = df[df["email"] == email]

    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)

    return df.head(limit).to_dict(orient="records")


def build_user_insights(email: str):
    if not DATA_PATH.exists():
        return {}

    df = pd.read_csv(DATA_PATH)
    if df.empty or "email" not in df.columns:
        return {}

    user_df = df[df["email"] == email]
    if user_df.empty:
        return {}

    mean_score = float(user_df["success_score"].mean())
    best_product = user_df.loc[user_df["success_score"].idxmax()].to_dict()
    worst_product = user_df.loc[user_df["success_score"].idxmin()].to_dict()

    per_category = []
    if "category" in user_df.columns:
        per_category = (
            user_df.groupby("category")["success_score"]
            .mean()
            .reset_index()
            .rename(columns={"success_score": "mean_score"})
            .sort_values("mean_score", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

    return {
        "mean_score": mean_score,
        "best_product": best_product,
        "worst_product": worst_product,
        "per_category": per_category,
    }

# ==================================================
# Dashboard from CSV
# ==================================================
def build_dashboard():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    if df.empty or "success_score" not in df.columns:
        return None

    overall_mean = float(df["success_score"].mean())

    if "category" in df.columns:
        per_cat = (
            df.groupby("category")["success_score"]
            .mean()
            .reset_index()
        )
        per_cat.rename(columns={"success_score": "mean_score"}, inplace=True)
        best_cat = per_cat.loc[per_cat["mean_score"].idxmax()].to_dict()
        worst_cat = per_cat.loc[per_cat["mean_score"].idxmin()].to_dict()
    else:
        per_cat = pd.DataFrame([])
        best_cat = {}
        worst_cat = {}

    best_product = df.loc[df["success_score"].idxmax()].to_dict()
    worst_product = df.loc[df["success_score"].idxmin()].to_dict()

    return {
        "overall_mean": overall_mean,
        "per_category": per_cat.to_dict(orient="records"),
        "best_category": best_cat,
        "worst_category": worst_cat,
        "best_product": best_product,
        "worst_product": worst_product,
    }

# ==================================================
# DEMO products enrichment
# ==================================================
def enrich_demo_products(products):
    enriched = []
    for p in products:
        price = float(p.get("price", 0))
        trend = float(p.get("trend_score", 50))
        success = predict_success(price, trend, p.get("category", "general"))
        risk = classify_risk(success)
        item = dict(p)
        item["success_score"] = success
        item["risk"] = risk
        enriched.append(item)
    return enriched

# ==================================================
# Amazon / AliExpress / TikTok (Fake + Scraping אם אפשר)
# ==================================================
def amazon_search_real(keyword: str):
    if not HAS_SCRAPER_DEPS or not keyword:
        return []

    try:
        url = f"https://www.amazon.com/s?k={keyword.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        products = []
        for item in soup.select(".s-result-item")[:10]:
            title = item.select_one("h2 span")
            price_whole = item.select_one(".a-price-whole")
            if not title or not price_whole:
                continue
            products.append(
                {
                    "title": title.text.strip(),
                    "price": float(price_whole.text.replace(",", "").replace(".", "")),
                }
            )
        return products
    except Exception:
        return []


def amazon_search_fake(keyword: str):
    keyword = (keyword or "").lower()
    fake_results = [
        {
            "title": "Wireless Earbuds Bluetooth 5.3",
            "price": 89,
            "image": "https://via.placeholder.com/150?text=Earbuds",
        },
        {
            "title": "Ring Light 18 inch",
            "price": 129,
            "image": "https://via.placeholder.com/150?text=Ring+Light",
        },
        {
            "title": "Yoga Mat Non-Slip",
            "price": 59,
            "image": "https://via.placeholder.com/150?text=Yoga+Mat",
        },
        {
            "title": "Car Vacuum Cleaner",
            "price": 149,
            "image": "https://via.placeholder.com/150?text=Car+Vacuum",
        },
    ]
    filtered = [p for p in fake_results if keyword in p["title"].lower()]
    return filtered or fake_results


def aliexpress_search_fake(keyword: str):
    keyword = (keyword or "").lower()
    fake_results = [
        {
            "title": "LED Room Light",
            "price": 25,
            "orders": 5400,
            "image": "https://via.placeholder.com/150?text=LED+Light",
        },
        {
            "title": "Mini Projector",
            "price": 199,
            "orders": 2100,
            "image": "https://via.placeholder.com/150?text=Projector",
        },
        {
            "title": "Phone Holder",
            "price": 19,
            "orders": 8700,
            "image": "https://via.placeholder.com/150?text=Phone+Holder",
        },
    ]
    filtered = [p for p in fake_results if keyword in p["title"].lower()]
    return filtered or fake_results


def tiktok_trends_fake(keyword: str = ""):
    keyword = (keyword or "").lower()
    fake_trends = [
        {"video": "LED Room Lights Aesthetic", "views": 1_200_000, "likes": 88_000},
        {"video": "Resistance Bands Workout", "views": 890_000, "likes": 74_000},
        {"video": "Car Phone Holder TikTok", "views": 540_000, "likes": 41_000},
        {"video": "Mini Projector Bedroom", "views": 760_000, "likes": 59_000},
    ]
    filtered = [v for v in fake_trends if keyword in v["video"].lower()]
    return filtered or fake_trends

# ==================================================
# השוואת שני מוצרים
# ==================================================
def compare_two_products(p1: dict, p2: dict) -> dict:
    for p in (p1, p2):
        p["success_score"], p["model_source"] = compute_success_score(
            p["price"], p["trend_score"], p.get("category", "general")
        )
        p["risk"] = classify_risk(p["success_score"])

    if p1["success_score"] > p2["success_score"]:
        winner = "p1"
        reason = "למוצר 1 יש Success גבוה יותר."
    elif p2["success_score"] > p1["success_score"]:
        winner = "p2"
        reason = "למוצר 2 יש Success גבוה יותר."
    else:
        if p1["price"] < p2["price"]:
            winner = "p1"
            reason = "ה-Success זהה, אבל מוצר 1 זול יותר."
        elif p2["price"] < p1["price"]:
            winner = "p2"
            reason = "ה-Success זהה, אבל מוצר 2 זול יותר."
        else:
            winner = "tie"
            reason = "שני המוצרים דומים מאוד במחיר וב-Success."

    return {"p1": p1, "p2": p2, "winner": winner, "reason": reason}

# ==================================================
# UI ROUTES
# ==================================================
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
        has_pytrends=HAS_PYTRENDS,
        has_model=has_model,
        model_info=model_info,
        train_message=train_message,
    )


@app.route("/train-model", methods=["POST"])
@login_required
def train_model_route():
    outcome = train_ml_model()
    if outcome.get("error"):
        session["train_message"] = f"❌ אימון נכשל: {outcome['error']}"
    else:
        info = outcome.get("info", {})
        r2 = info.get("r2_test")
        mae = info.get("mae_test")
        session["train_message"] = (
            "✅ מודל ML חינמי אומן בהצלחה"
            + (f" | R²: {r2:.3f} MAE: {mae:.2f}" if r2 is not None and mae is not None else "")
        )
    return redirect(url_for("index"))


@app.route("/export-history")
@login_required
def export_history():
    email = session["user"]["email"]
    predictions = load_predictions(email, limit=500)
    if not predictions:
        return json_response({"error": "אין נתונים לייצוא עבור המשתמש"}, 404)

    df = pd.DataFrame(predictions)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    filename = f"aicommerce_history_{email.replace('@', '_')}.csv"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(buffer.getvalue(), mimetype="text/csv", headers=headers)


@app.route("/shop")
@login_required
def shop():
    email = session["user"]["email"]
    plan = session["user"]["plan"]
    used_today = get_user_daily_usage(email)
    limit_today = FREE_DAILY_LIMIT if plan == "FREE" else None

    products = enrich_demo_products(DEMO_PRODUCTS)
    best_product = max(products, key=lambda p: p["success_score"])

    return render_template(
        "shop.html",
        user=session["user"],
        used_today=used_today,
        limit_today=limit_today,
        products=products,
        best_product=best_product,
    )

# ---------- Auth ----------


@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()
    error = None

    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        if email in users and users[email]["password"] == pw:
            session["user"] = {
                "email": email,
                "plan": users[email].get("plan", "FREE"),
            }
            return redirect(url_for("index"))
        else:
            error = "אימייל או סיסמה לא נכונים"

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    users = load_users()
    error = None
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw = request.form["password"]

        if not email or not pw:
            error = "חובה למלא אימייל וסיסמה"
        elif email in users:
            error = "אימייל כבר קיים במערכת"
        else:
            users[email] = {
                "password": pw,
                "plan": "FREE",
                "api_key": generate_api_key(),
            }
            save_users(users)
            return redirect(url_for("login"))

    return render_template("register.html", error=error)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    users = load_users()
    msg = None
    error = None

    if request.method == "POST":
        email = request.form["email"].strip().lower()
        new_pw = request.form["new_password"]
        if email in users:
            users[email]["password"] = new_pw
            save_users(users)
            msg = "הסיסמה עודכנה, אפשר להתחבר עם הסיסמה החדשה."
        else:
            error = "אימייל לא נמצא במערכת."

    return render_template("forgot.html", success=msg, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/upgrade")
@login_required
def upgrade():
    users = load_users()
    email = session["user"]["email"]
    if email in users:
        users[email]["plan"] = "PRO"
        save_users(users)
        session["user"]["plan"] = "PRO"
    return redirect(url_for("index"))


@app.route("/my-api-key")
@login_required
def my_api_key():
    users = load_users()
    email = session["user"]["email"]
    if email not in users:
        return json_response({"error": "User not found"}, 404)
    if "api_key" not in users[email]:
        users[email]["api_key"] = generate_api_key()
        save_users(users)
    return json_response({"email": email, "api_key": users[email]["api_key"]})

# ---------- Predict (UI) ----------


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    email = session["user"]["email"]
    plan = session["user"]["plan"]

    used_today = get_user_daily_usage(email)
    if plan == "FREE" and used_today >= FREE_DAILY_LIMIT:
        session["last_result"] = {
            "error": "הגעת למגבלת FREE להיום. שדרג ל-PRO כדי להמשיך להשתמש."
        }
        return redirect(url_for("index"))

    name = request.form["name"].strip()
    category = request.form["category"].strip()
    price = float(request.form["price"])
    trend_input = request.form.get("trend_score", "").strip()

    if trend_input == "" or trend_input.lower() == "auto":
        trend_score = get_trend_from_google(name or category)
        trend_source = "google"
    else:
        trend_score = float(trend_input)
        trend_source = "manual"

    success_score, model_source = compute_success_score(price, trend_score, category)
    risk = classify_risk(success_score)
    created_at = datetime.now().isoformat(timespec="seconds")

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                email,
                name,
                category,
                price,
                trend_score,
                success_score,
                risk,
                created_at,
            ]
        )

    session["last_result"] = {
        "name": name,
        "category": category,
        "price": price,
        "trend_score": trend_score,
        "trend_source": trend_source,
        "success_score": success_score,
        "risk": risk,
        "model_source": model_source,
    }

    maybe_autotrain_model()

    return redirect(url_for("index"))

# ---------- Compare (UI) ----------


@app.route("/compare", methods=["POST"])
@login_required
def compare():
    name1 = request.form["c_name1"].strip()
    category1 = request.form["c_category1"].strip()
    price1 = float(request.form["c_price1"])
    trend1_in = request.form.get("c_trend1", "").strip()

    name2 = request.form["c_name2"].strip()
    category2 = request.form["c_category2"].strip()
    price2 = float(request.form["c_price2"])
    trend2_in = request.form.get("c_trend2", "").strip()

    if trend1_in == "" or trend1_in.lower() == "auto":
        trend1 = get_trend_from_google(name1 or category1)
    else:
        trend1 = float(trend1_in)

    if trend2_in == "" or trend2_in.lower() == "auto":
        trend2 = get_trend_from_google(name2 or category2)
    else:
        trend2 = float(trend2_in)

    p1 = {"name": name1, "category": category1, "price": price1, "trend_score": trend1}
    p2 = {"name": name2, "category": category2, "price": price2, "trend_score": trend2}

    compare_result = compare_two_products(p1, p2)
    session["compare_result"] = compare_result

    return redirect(url_for("index"))

# ==================================================
# API Routes
# ==================================================


@app.route("/api/predict", methods=["POST"])
@api_key_required
def api_predict(api_user):
    data = request.get_json(force=True)
    price = float(data.get("price", 0))
    category = data.get("category", "general")
    keyword = data.get("keyword", "")

    trend_score = get_trend_from_google(keyword or category)
    success_score, model_source = compute_success_score(price, trend_score, category)
    risk = classify_risk(success_score)

    created_at = datetime.now().isoformat(timespec="seconds")
    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                api_user["email"],
                keyword or "API Product",
                category,
                price,
                trend_score,
                success_score,
                risk,
                created_at,
            ]
        )

    maybe_autotrain_model()

    return json_response(
        {
            "success_score": success_score,
            "risk": risk,
            "trend_used": trend_score,
            "plan": api_user["plan"],
            "model_source": model_source,
        }
    )


@app.route("/api/history", methods=["GET"])
@api_key_required
def api_history(api_user):
    limit = int(request.args.get("limit", 20))
    predictions = load_predictions(api_user["email"], limit=limit)
    return json_response({"items": predictions, "count": len(predictions)})


@app.route("/api/top-products", methods=["GET"])
@api_key_required
def api_top_products(api_user):
    if not DATA_PATH.exists():
        return json_response([])

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return json_response([])

    top = df.sort_values("success_score", ascending=False).head(10)
    return json_response(top.to_dict(orient="records"))


@app.route("/api/model-status", methods=["GET"])
@api_key_required
def api_model_status(api_user):
    info = get_model_info()
    has_model = load_trained_model() is not None
    fallback = "Rule-based MVP model (trend + price)" if not has_model else None

    return json_response(
        {
            "has_model": has_model,
            "info": info or fallback,
            "plan": api_user["plan"],
        }
    )


@app.route("/api/amazon-search", methods=["GET"])
@api_key_required
def api_amazon_search(api_user):
    keyword = request.args.get("keyword", "")
    results = amazon_search_real(keyword)
    if not results:
        results = amazon_search_fake(keyword)
    return json_response({"source": "amazon", "keyword": keyword, "results": results})


@app.route("/api/aliexpress-search", methods=["GET"])
@api_key_required
def api_aliexpress_search(api_user):
    keyword = request.args.get("keyword", "")
    results = aliexpress_search_fake(keyword)
    return json_response(
        {
            "source": "aliexpress",
            "keyword": keyword,
            "results": results,
        }
    )


@app.route("/api/tiktok-trends", methods=["GET"])
@api_key_required
def api_tiktok_trends(api_user):
    keyword = request.args.get("keyword", "")
    trends = tiktok_trends_fake(keyword)
    return json_response({"source": "tiktok", "keyword": keyword, "trends": trends})


@app.route("/api/winning-product", methods=["GET"])
@api_key_required
def api_winning_product(api_user):
    dash = build_dashboard()
    if not dash:
        return json_response({"error": "No data yet"}, 404)
    return json_response(
        {"winning_product": dash["best_product"], "reason": "Highest Success Score"}
    )


@app.route("/api/live-dashboard", methods=["GET"])
@api_key_required
def api_live_dashboard(api_user):
    dash = build_dashboard()
    if not dash:
        return json_response({})
    return json_response(dash)


@app.route("/api/demo-products", methods=["GET"])
@api_key_required
def api_demo_products(api_user):
    return json_response(enrich_demo_products(DEMO_PRODUCTS))


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return json_response(
        {
            "info": "AICommerce external API – MVP Demo",
            "auth": "Use X-API-Key header or api_key query/body",
            "endpoints": {
                "POST /api/predict": "Predict success for a product",
                "GET /api/top-products": "Top products by success_score",
                "GET /api/model-status": "Simple model status",
                "GET /api/amazon-search": "Search Amazon (scraper + fake fallback)",
                "GET /api/aliexpress-search": "Search AliExpress (fake)",
                "GET /api/tiktok-trends": "TikTok trending videos (fake)",
                "GET /api/winning-product": "Best product in the system",
                "GET /api/live-dashboard": "Global dashboard stats",
                "GET /api/demo-products": "Demo store products",
            },
        }
    )

# ==================================================
# Shopify Webhook – לוג בלבד
# ==================================================


@app.route("/webhook/shopify/product-created", methods=["POST"])
def shopify_product_created():
    payload = request.get_json(force=True)
    with open(SHOPIFY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return "", 200


@app.route("/train-model", methods=["POST"])
@login_required
def train_model_route():
    session["last_result"] = {
        "error": "אימון מודל אמיתי עדיין לא זמין בגרסת MVP"
    }
    return redirect(url_for("index"))


@app.route("/export-history")
@login_required
def export_history():
    if not DATA_PATH.exists():
        return json_response({"error": "No data to export"}, 404)

    return Response(
        open(DATA_PATH, "rb"),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=history.csv"},
    )


# ==================================================
# הרצה
# ==================================================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
