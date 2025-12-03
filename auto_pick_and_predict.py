import json
import pandas as pd
import joblib
import random

MODEL = joblib.load("aicommerce_model.pkl")

with open("market_products.json", "r", encoding="utf-8") as f:
    PRODUCTS = json.load(f)

def predict_with_ml(price, trend_score, category, orders_now):
    df = pd.DataFrame([{
        "price": price,
        "trend_score": trend_score,
        "category": category,
        "orders_now": orders_now
    }])

    proba = MODEL.predict_proba(df)[0][1]
    return int(proba * 100)

results = []

for p in PRODUCTS:
    score = predict_with_ml(
        price=p["price"],
        trend_score=random.randint(50, 90),
        category=p["category"],
        orders_now=p["orders_now"]
    )

    p["future_success_probability"] = score
    results.append(p)

results = sorted(results, key=lambda x: x["future_success_probability"], reverse=True)

print("\n🔥 Top Winning Products (Auto Pick):\n")
for r in results:
    print(
        f"✅ {r['name']} | ₪{r['price']} | Orders: {r['orders_now']} | "
        f"Score: {r['future_success_probability']}%"
    )
