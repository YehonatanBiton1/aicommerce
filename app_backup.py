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
import os
import joblib



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
@app.route("/auto-pick")
@login_required
def auto_pick():

    token = os.getenv("EBAY_ACCESS_TOKEN")
    if not token:
        return render_template("auto_pick.html", results=[], error="❌ חסר EBAY_ACCESS_TOKEN")

    keywords = [
        "wireless charger",
        "phone accessory",
        "car gadget",
        "desk organizer",
        "led light"
    ]

    query = random.choice(keywords)

    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={query}&limit=50"

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        products = data.get("itemSummaries", [])
        print("✅ TOTAL FROM EBAY:", len(products))
    except Exception as e:
        print("❌ EBAY ERROR:", e)
        products = []

    blocked_words = [
        "perfume", "fragrance", "cologne",
        "shoes", "sneakers", "boots",
        "laptop", "computer", "notebook",
        "camera", "dslr",
        "sony", "canon", "dell", "hp"
    ]

    results = []

    for p in products:

        title = p.get("title", "Unknown")
        title_lower = title.lower()
        price = float(p.get("price", {}).get("value", 0))

        # ✅ חסימת מותגים בעייתיים
        if any(word in title_lower for word in blocked_words):
            continue

        # ✅ טווח מחיר ריאלי לדרופשיפינג
        if price < 12 or price > 50:
            continue

        # ✅ חישוב רווח
        markup = random.uniform(1.4, 2.2)   # מכפיל משתנה אמיתי
        sale_price = round(price * markup, 2)
        profit = round(sale_price - price, 2)
        roi = round((profit / price) * 100, 1)
        if profit < 8:
            continue

        roi = round((profit / price) * 100, 1)

        orders_now = random.randint(30, 200)
        future_prob = random.randint(55, 90)

        result = {
            "title": title,
            "price": price,
            "sale_price": sale_price,
            "profit": profit,
            "roi": roi,
            "category": "dropshipping",
            "orders_now": orders_now,
            "future_success_probability": future_prob,
            "link": p.get("itemWebUrl") or "https://www.ebay.com",
            "image": p.get("image", {}).get("imageUrl", "")
        }

        results.append(result)

        # ✅ עוצרים ב־20 מוצרים איכותיים
        if len(results) >= 20:
            break

    return render_template("auto_pick.html", results=results)

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
def export_auto_pick_csv():
    import csv
    from flask import Response

    with open("market_products.json", encoding="utf-8") as f:
        products = json.load(f)

    def generate():
        header = "title,category,price,orders_now,future_success_probability,link\n"
        yield header

        for p in products:
            title = p.get("title","").replace(",", " ")
            category = p.get("category","")
            price = p.get("price","")
            orders = p.get("orders_now","")
            prob = p.get("future_success_probability","")
            link = p.get("link","")

            yield f"{title},{category},{price},{orders},{prob},{link}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=auto_pick.csv"}
    )

from flask import request, jsonify
import json

@app.route("/api/chat", methods=["POST"])
@login_required
def chat_api():
    data = request.json
    user_message = data.get("message", "").lower()

    # ========================
    # ✅ שמירת לידים - חייב להיות בהתחלה
    # ========================

    if "@" in user_message and "." in user_message:
        try:
            with open("leads.json", "r", encoding="utf-8") as f:
                leads = json.load(f)
        except:
            leads = []

        leads.append({"email": user_message})

        with open("leads.json", "w", encoding="utf-8") as f:
            json.dump(leads, f, ensure_ascii=False, indent=2)

        return jsonify({"reply": "✅ תודה! האימייל נשמר ונחזור אליך עם הצעות רווחיות."})

    if user_message.isdigit() and len(user_message) >= 9:
        try:
            with open("leads.json", "r", encoding="utf-8") as f:
                leads = json.load(f)
        except:
            leads = []

        leads.append({"phone": user_message})

        with open("leads.json", "w", encoding="utf-8") as f:
            json.dump(leads, f, ensure_ascii=False, indent=2)

        return jsonify({"reply": "✅ מספר הטלפון נשמר! נציג יחזור אליך בקרוב."})

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
