# ==================================================
# AICommerce – גרסת MVP חזקה ומאוחדת (חינמית)
# ==================================================

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import csv
import csv
import json
import secrets
import random
from datetime import datetime, date
from pathlib import Path
from functools import wraps
import pandas as pd
import joblib
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")



EBAY_MARKETPLACE_ID = "EBAY_US"

DROPSHIPPING_KEYWORDS = [
    "wireless charger",
    "phone accessory",
    "car gadget",
    "desk organizer",
    "led strip",
    "home gadget"
]

BLOCKED_WORDS = [
    "perfume", "fragrance", "cologne",
    "shoes", "sneakers", "boots",
    "laptop", "computer", "notebook",
    "camera", "dslr",
    "sony", "canon", "dell", "hp"
]

MIN_PRICE = 10
MAX_PRICE = 60
MIN_PROFIT = 8
MIN_ROI = 40
TOP_N = 15


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
conversations = {}
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "aicommerce_data.csv"
USERS_PATH = BASE_DIR / "users.json"
API_USAGE_PATH = BASE_DIR / "api_usage.json"
SHOPIFY_LOG_PATH = BASE_DIR / "shopify_webhook_log.jsonl"

FREE_DAILY_LIMIT = 10
FREE_API_DAILY_LIMIT = 100

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

    if api_user.get("plan", "FREE") != "FREE":
        entry["count"] += 1
        usage[key] = entry
        save_api_usage(usage)
        return True

    if entry["count"] >= FREE_API_DAILY_LIMIT:
        return False

    entry["count"] += 1
    usage[key] = entry
    save_api_usage(usage)
    return True

# ==================================================
# Decorators ✅ חייבים להיות לפני ROUTES
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
            return Response(
                json.dumps({"error": "Missing API key"}, ensure_ascii=False),
                status=401,
                mimetype="application/json"
            )

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
            return Response(
                json.dumps({"error": "Invalid API key"}, ensure_ascii=False),
                status=401,
                mimetype="application/json"
            )

        if not check_api_limit(api_user):
            return Response(
                json.dumps({"error": "Daily API limit reached (FREE)"}, ensure_ascii=False),
                status=429,
                mimetype="application/json"
            )

        kwargs["api_user"] = api_user
        return fn(*args, **kwargs)

    return wrapper
# ==================================================
# מוצרי DEMO לחנות
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
        "description": "אוזניות בלוטות' עם ביטול רעשים."
    },
    {
        "id": 3,
        "name": "Ring Light + Tripod",
        "category": "יוצרי תוכן",
        "price": 139.0,
        "trend_score": 88,
        "image": "https://images.pexels.com/photos/6898859/pexels-photo-6898859.jpeg",
        "description": "סט צילום מקצועי."
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
# AI Engine – Google Trends + Success Score
# ==================================================
def get_trend_from_google(keyword: str) -> float:
    keyword = (keyword or "").strip()
    if not keyword:
        return random.randint(40, 80)

    if not HAS_PYTRENDS:
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
# ML ENGINE
# ==================================================
MODEL = None
MODEL_INFO = {}

if os.path.exists("aicommerce_model.pkl"):
    try:
        MODEL = joblib.load("aicommerce_model.pkl")
        with open("aicommerce_model_info.json", "r", encoding="utf-8") as f:
            MODEL_INFO = json.load(f)
        print("✅ מודל ML נטען בהצלחה")
    except Exception as e:
        print("❌ שגיאה בטעינת מודל:", e)
        MODEL = None

def predict_with_ml(price, trend_score, category):
    if not MODEL:
        return None

    df = pd.DataFrame([{
        "price": float(price),
        "trend_score": float(trend_score),
        "category": category
    }])

    proba = MODEL.predict_proba(df)[0][1]
    return int(proba * 100)

def predict_with_ml_real(price, trend_score, category, orders_now):
    if not MODEL:
        return None

    df = pd.DataFrame([{
        "price": float(price),
        "trend_score": float(trend_score),
        "category": category,
        "orders_now": int(orders_now)
    }])

    proba = MODEL.predict_proba(df)[0][1]
    return int(proba * 100)

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
        per_cat = df.groupby("category")["success_score"].mean().reset_index()
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
# Fake Scrapers
# ==================================================
def amazon_search_fake(keyword: str):
    return [
        {"title": "Wireless Earbuds", "price": 89},
        {"title": "Ring Light", "price": 129},
    ]

def aliexpress_search_fake(keyword: str):
    return [
        {"title": "LED Light", "price": 25, "orders": 5400},
        {"title": "Mini Projector", "price": 199, "orders": 2100},
    ]

def tiktok_trends_fake(keyword: str = ""):
    return [
        {"video": "LED Lights Aesthetic", "views": 1200000},
        {"video": "Resistance Bands", "views": 890000},
    ]

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
        reason = "מוצר 1 עדיף"
    elif p2["success_score"] > p1["success_score"]:
        winner = "p2"
        reason = "מוצר 2 עדיף"
    else:
        winner = "tie"
        reason = "שוויון"

    return {"p1": p1, "p2": p2, "winner": winner, "reason": reason}
# ==================================================
# UI ROUTES
# ==================================================

@app.route("/")
@login_required
def index():
    email = session["user"]["email"]
    plan = session["user"]["plan"]

    used_today = 0
    limit_today = FREE_DAILY_LIMIT if plan == "FREE" else None

    history = []
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        if not df.empty and "email" in df.columns and "category" in df.columns:
            user_df = df[df["email"] == email]

            # ✅ סינון קטגוריות בעייתיות מהתצוגה
            blocked_categories = [
                "shoes", "sneakers", "boots",
                "perfume", "fragrance", "cologne",
                "laptop", "computer", "pc", "notebook",
                "camera", "dslr"
            ]

            def is_allowed(category):
                category = str(category).lower()
                for blocked in blocked_categories:
                    if blocked in category:
                        return False
                return True

            user_df = user_df[user_df["category"].apply(is_allowed)]

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
        has_model=MODEL is not None,
        model_info=MODEL_INFO if MODEL else {},
    )

# ==================================================
# SHOP
# ==================================================

@app.route("/shop")
@login_required
def shop():
    email = session["user"]["email"]
    plan = session["user"]["plan"]

    products = enrich_demo_products(DEMO_PRODUCTS)
    best_product = max(products, key=lambda p: p["success_score"])

    return render_template(
        "shop.html",
        user=session["user"],
        products=products,
        best_product=best_product,
        used_today=0,
        limit_today=None if plan != "FREE" else FREE_DAILY_LIMIT,
    )

# ==================================================
# AUTH
# ==================================================

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


# ==================================================
# PREDICT (UI)
# ==================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    email = session["user"]["email"]
    plan = session["user"]["plan"]

    name = request.form["name"].strip()
    category = request.form["category"].strip()

    # ✅ חסימת קטגוריות בעייתיות
    blocked_categories = [
        "shoes", "sneakers", "boots",
        "perfume", "fragrance", "cologne",
        "laptop", "computer", "pc", "notebook",
        "refurbished", "used"
    ]

    category_lower = category.lower()

    for blocked in blocked_categories:
        if blocked in category_lower:
            session["last_result"] = {
                "name": name,
                "category": category,
                "price": 0,
                "trend_score": 0,
                "trend_source": "blocked",
                "success_score": 0,
                "risk": "HIGH",
                "model_source": "blocked"
            }
            return redirect(url_for("index"))

    # ✅ ממשיכים רק אם לא נחסם
    price = float(request.form["price"])
    trend_input = request.form.get("trend_score", "").strip()

    if trend_input == "" or trend_input.lower() == "auto":
        trend_score = get_trend_from_google(name or category)
        trend_source = "google"
    else:
        trend_score = float(trend_input)
        trend_source = "manual"

    ml_score = predict_with_ml(price, trend_score, category)

    if ml_score is not None:
        success_score = ml_score
        model_source = "ml"
    else:
        success_score = predict_success(price, trend_score, category)
        model_source = "rules"

    risk = classify_risk(success_score)
    created_at = datetime.now().isoformat(timespec="seconds")

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            email,
            name,
            category,
            price,
            trend_score,
            success_score,
            risk,
            created_at,
        ])

    session["last_result"] = {
        "name": name,
        "category": category,
        "price": price,
        "trend_score": trend_score,
        "trend_source": trend_source,
        "success_score": success_score,
        "risk": risk,
        "model_source": model_source
    }

    return redirect(url_for("index"))

# ==================================================
# COMPARE
# ==================================================
def get_ebay_app_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    response = requests.post(
        url,
        headers=headers,
        data=data,
        auth=(EBAY_CLIENT_ID, EBAY_CLIENT_SECRET)
    )

    print("🔐 EBAY TOKEN RESPONSE STATUS:", response.status_code)
    print("🔐 EBAY TOKEN FULL RESPONSE:", response.text)

    token_data = response.json()

    if "access_token" not in token_data:
        print("❌ EBAY TOKEN ERROR – access_token missing")
        return None

    return token_data["access_token"]

@app.route("/compare", methods=["POST"])
@login_required
def compare():
    p1 = {
        "name": request.form["c_name1"],
        "category": request.form["c_category1"],
        "price": float(request.form["c_price1"]),
        "trend_score": get_trend_from_google(request.form["c_name1"])
    }

    p2 = {
        "name": request.form["c_name2"],
        "category": request.form["c_category2"],
        "price": float(request.form["c_price2"]),
        "trend_score": get_trend_from_google(request.form["c_name2"])
    }

    session["compare_result"] = compare_two_products(p1, p2)
    return redirect(url_for("index"))

# ==================================================
# AUTO PICK ✅ מה שהמשקיע רצה
# ==================================================
token = get_ebay_app_token()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
def fetch_ebay_products(token, keywords):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    all_products = []

    for keyword in keywords:
        url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={keyword}&limit=30"

        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"🔍 keyword='{keyword}' STATUS:", response.status_code)

            if response.status_code != 200:
                continue

            data = response.json()
            items = data.get("itemSummaries", [])
            print("📦 ITEMS FROM EBAY:", len(items))

            for item in items:
                try:
                    price = float(item.get("price", {}).get("value", 0))

                    image_data = item.get("image")
                    if isinstance(image_data, dict):
                        image_url = image_data.get("imageUrl")
                    elif isinstance(image_data, str):
                        image_url = image_data
                    else:
                        image_url = None

                    seller = item.get("seller", {})

                    product = {
                        "title": item.get("title"),
                        "price": price,
                        "image": image_url,
                        "link": item.get("itemWebUrl"),
                        "category": item.get("categories", [{}])[0].get("categoryName"),
                        "seller_rating": float(seller.get("feedbackPercentage", 0)),
                        "seller_score": int(seller.get("feedbackScore", 0)),
                    }

                    all_products.append(product)

                except Exception as e:
                    print("⚠️ ITEM SKIPPED:", e)

        except Exception as e:
            print("❌ EBAY FETCH ERROR:", e)

    print("✅ TOTAL RAW PRODUCTS:", len(all_products))
    return all_products

def get_ebay_app_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"

    client_id = os.getenv("EBAY_CLIENT_ID")
    client_secret = os.getenv("EBAY_CLIENT_SECRET")

    auth = (client_id, client_secret)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    response = requests.post(url, headers=headers, data=data, auth=auth)

    if response.status_code != 200:
        print("❌ FAILED TO GET APP TOKEN:", response.text)
        return None

    token = response.json().get("access_token")
    print("✅ APP TOKEN RECEIVED")
    return token

def clean_and_filter_raw_products(raw_products):
    cleaned = []

    for p in raw_products:
        price = float(p.get("price", 0))

        if price <= 0:
            continue

        cleaned.append(p)

    print("✅ AFTER BASIC FILTERS:", len(cleaned))
    return cleaned


import random

def enrich_product_metrics(product):
    try:
        price = product.get("price")

        # אם המחיר לא מספר – נופל ל־0
        try:
            price = float(str(price).replace("$", "").strip())
        except:
            price = 0.0

        # מחיר מכירה – סימולציה ריאלית
        markup = random.uniform(1.6, 2.2)
        sale_price = round(price * markup, 2)

        profit = round(sale_price - price, 2)

        if price > 0:
            roi = round((profit / price) * 100, 1)
        else:
            roi = 0

        orders_now = random.randint(20, 200)

        product["price"] = price
        product["sale_price"] = sale_price
        product["profit"] = profit
        product["roi"] = roi
        product["orders_now"] = orders_now

        print("✅ ENRICH:", price, sale_price, profit, roi, orders_now)

        return product

    except Exception as e:
        print("❌ ENRICH ERROR:", e)
        product["sale_price"] = 0
        product["profit"] = 0
        product["roi"] = 0
        product["orders_now"] = 0
        return product


def safe_float(val):
    try:
        if val is None:
            return 0.0
        if isinstance(val, str):
            val = val.replace("$", "").replace("%", "").strip()
        return float(val)
    except:
        return 0.0


def safe_int(val):
    try:
        return int(float(val))
    except:
        return 0


def score_product(product):
    try:
        roi = float(product.get("roi", 0))
        profit = float(product.get("profit", 0))
        orders = int(product.get("orders_now", 0))

        roi = max(min(roi, 150), 0)        # 0–150
        profit = max(min(profit, 50), 0)  # 0–50$
        orders = max(min(orders, 300), 0) # 0–300

        success_rate = (
            (roi / 150) * 40 +
            (profit / 50) * 35 +
            (orders / 300) * 25
        )

        success_rate = round(success_rate, 1)

        product["success_rate"] = success_rate
        product["winner_score"] = success_rate

        return product

    except Exception as e:
        print("❌ SCORE ERROR:", e)
        product["success_rate"] = 0
        product["winner_score"] = 0
        return product


def map_to_view_model(p):
    return {
        "title": p.get("title"),
        "category": p.get("category"),
        "price": p.get("price"),
        "sale_price": p.get("sale_price"),
        "profit": p.get("profit"),
        "roi": p.get("roi"),
        "orders_now": p.get("orders_now"),
        "image": p.get("image"),
        "url": p.get("url") or p.get("itemHref") or p.get("link"),

        # ✅ זה הקריטי
        "future_success_probability": float(p.get("success_rate", 0))
    }



@app.route("/auto-pick")
@login_required
def auto_pick():
    try:
        token = get_ebay_app_token()

        raw_products = fetch_ebay_products(token, DROPSHIPPING_KEYWORDS)
        print(f"📦 RAW PRODUCTS: {len(raw_products)}")

        cleaned_products = clean_and_filter_raw_products(raw_products)
        print(f"🧹 AFTER CLEAN: {len(cleaned_products)}")

        enriched = [enrich_product_metrics(p) for p in cleaned_products]
        scored = [score_product(p) for p in enriched]

        # ✅ מיון לפי אחוז הצלחה
        scored.sort(key=lambda p: p.get("success_rate", 0), reverse=True)

        # ✅ שמירה של *כל* המוצרים ל־CSV
        ALL_RESULTS = [map_to_view_model(p) for p in scored]

        with open("market_products.json", "w", encoding="utf-8") as f:
            json.dump(ALL_RESULTS, f, ensure_ascii=False, indent=4)

        # ✅ רק ה־TOP יוצגו במסך
        top_products = scored[:TOP_N]
        results = [map_to_view_model(p) for p in top_products]

        # ✅ שמירה של הטופ בלבד (לא חובה)
        with open("auto_pick_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"🏆 FINAL WINNERS: {len(results)}")

        return render_template("auto_pick.html", results=results)

    except Exception as e:
        print("🔥 AUTO PICK CRASH:", e)
        return render_template("auto_pick.html", results=[], error=str(e))

# ==================================================
# API
# ==================================================

@app.route("/api/demo-products")
@api_key_required
def api_demo_products(api_user):
    return json_response(enrich_demo_products(DEMO_PRODUCTS))

# ==================================================
# EXPORT
# ==================================================

@app.route("/export-history")
@login_required
def export_history():
    return Response(
        open(DATA_PATH, "rb"),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=history.csv"},
    )
@app.route("/train-model", methods=["POST"])
@login_required
def train_model_route():
    session["last_result"] = {
        "error": "אימון מודל אמיתי עדיין לא זמין בגרסת MVP"
    }
    return redirect(url_for("index"))

@app.route("/export_auto_pick_csv")
@app.route("/export_auto_pick_csv")
def export_auto_pick_csv():
    import csv
    from flask import Response
    import json

    with open("market_products.json", encoding="utf-8") as f:
        products = json.load(f)

    def generate():
        header = "title,category,price,orders_now,success_rate,link\n"
        yield header

        for p in products:
            title = p.get("title", "").replace(",", " ")
            category = p.get("category", "")
            price = p.get("price", "")
            orders = p.get("orders_now", "")
            prob = p.get("success_rate", "")
            link = p.get("url", "")

            yield f"{title},{category},{price},{orders},{prob},{link}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=auto_pick.csv"}
    )

    # ========================
    # ✅ טעינת מוצרים
    # ========================

    try:
        with open("market_products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
    except:
        products = []

    if not products:
        return jsonify({"reply": "❌ אין כרגע נתוני מוצרים במערכת."})

    # ========================
    # ✅ חיפוש מוצר רווחי
    # ========================

    if "מוצר" in user_message or "רווחי" in user_message:
        top = sorted(products, key=lambda x: x.get("orders_now", 0), reverse=True)[:3]
        reply = "🔥 הנה 3 מוצרים חזקים כרגע:\n"
        for p in top:
            reply += f"- {p['title']} | 💰 ₪{p['price']}\n"
        return jsonify({"reply": reply})

    # ========================
    # ✅ חיפוש לפי קטגוריה
    # ========================

    for p in products:
        if p.get("category", "").lower() in user_message:
            return jsonify({
                "reply": f"✅ מצאתי מוצר בקטגוריה שביקשת:\n{p['title']} – ₪{p['price']}"
            })

    # ========================
    # ✅ ברירת מחדל
    # ========================

    return jsonify({
        "reply": "🤖 אני יכול להמליץ על מוצרים רווחיים, לחפש לפי קטגוריה, או לשמור אימייל וטלפון."
    })

@app.route("/api/chat", methods=["POST"])
@login_required
def chat_api():
    data = request.json
    user_message = data.get("message", "")
    user_email = session["user"]["email"]

    if user_email not in conversations:
        conversations[user_email] = []

    # ✅ הוראות למודל
    instructions = """
אתה עוזר AI מומחה לדרופשיפינג.
אתה עוזר למצוא מוצרים מנצחים מאיביי בלבד.
אתה מחזיר:
שם מוצר, קישור לאיביי, מחיר קנייה, מחיר מכירה ורווח.
אתה מדבר עברית בלבד.
"""

    # ✅ זיכרון שיחה
    history_text = ""
    for msg in conversations[user_email]:
        history_text += f"{msg['role']}: {msg['content']}\n"

    full_input = history_text + f"user: {user_message}"

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": "gemma:2b",
                "prompt": instructions + "\n" + full_input,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            print("❌ OLLAMA BAD STATUS:", response.status_code, response.text)
            return jsonify({"reply": "❌ המודל לא החזיר תשובה תקינה"})

        result = response.json()
        reply = result.get("response", "").strip()

        if not reply:
            reply = "❌ המודל החזיר תשובה ריקה"

    except Exception as e:
        print("❌ OLLAMA CONNECTION ERROR:", e)
        return jsonify({"reply": "❌ שגיאה בחיבור ל־Ollama"})

    # ✅ שמירת זיכרון
    conversations[user_email].append({"role": "user", "content": user_message})
    conversations[user_email].append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply})



@app.route("/chat")
@login_required
def chat_page():
    return render_template("chat.html")

import os
import requests

#===============================================
# RUN
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
