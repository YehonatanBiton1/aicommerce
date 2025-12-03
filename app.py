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
from datetime import datetime, date
from pathlib import Path
from functools import wraps

import pandas as pd

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
        p["success_score"] = predict_success(
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

    history = []
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if not df.empty and "email" in df.columns:
            user_df = df[df["email"] == email]
            history = user_df.tail(20).to_dict(orient="records")

    dashboard = build_dashboard()
    result = session.pop("last_result", None)
    compare_result = session.pop("compare_result", None)

    return render_template(
        "index.html",
        user=session["user"],
        used_today=used_today,
        limit_today=limit_today,
        history=history,
        dashboard=dashboard,
        result=result,
        compare_result=compare_result,
        has_pytrends=HAS_PYTRENDS,
        has_model=False,
        model_info={},
    )


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

    success_score = predict_success(price, trend_score, category)
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
    }

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
    success_score = predict_success(price, trend_score, category)
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

    return json_response(
        {
            "success_score": success_score,
            "risk": risk,
            "trend_used": trend_score,
            "plan": api_user["plan"],
        }
    )


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
    return json_response(
        {
            "has_model": False,
            "info": "Rule-based MVP model (trend + price)",
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
        "error": "🚧 אימון מודל אמיתי עדיין לא זמין בגרסת MVP"
    }
    return redirect(url_for("index"))

# ==================================================
# הרצה
# ==================================================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
