import csv
import pandas as pd
import joblib

# טעינת המודל שאימנו קודם
model = joblib.load("aicommerce_model.pkl")


def predict_success_ml(product):
    """
    מקבל dict של מוצר, מחזיר Success Score 0-100 בעזרת המודל
    """
    df = pd.DataFrame([{
        "price": product["price"],
        "trend_score": product["trend_score"],
        "category": product["category"]
    }])

    score = model.predict(df)[0]

    # הגבלה ל-0–100
    score = max(0, min(100, int(score)))
    return score


def classify_risk(score):
    """
    מחזיר רמת סיכון לפי הציון
    """
    if score >= 70:
        return "פוטנציאל גבוה"
    elif score >= 40:
        return "בינוני"
    else:
        return "סיכון גבוה"


print("=== AICommerce - חיזוי הצלחת מוצר (מודל ML) ===")

name = input("שם מוצר: ")
category = input("קטגוריה: ")

# קבלת מחיר כמספר
while True:
    price_input = input("מחיר (₪): ")
    try:
        price = float(price_input)
        break
    except ValueError:
        print("❌ חייב להכניס מספר למחיר")

# קבלת ציון טרנד כמספר
while True:
    trend_input = input("ציון טרנד 0-100: ")
    try:
        trend_score = float(trend_input)
        break
    except ValueError:
        print("❌ חייב להכניס מספר בין 0 ל-100")

product = {
    "name": name,
    "category": category,
    "price": price,
    "trend_score": trend_score
}

# חיזוי בעזרת המודל
success_score = predict_success_ml(product)
risk = classify_risk(success_score)

print("\n------ תוצאה ------")
print("שם המוצר:", name)
print("Success Score (ML):", success_score)
print("רמת סיכון:", risk)

# שמירת התוצאה לקובץ הדאטה
with open("aicommerce_data.csv", mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    if file.tell() == 0:
        writer.writerow(["name", "category", "price", "trend_score", "success_score", "risk"])

    writer.writerow([name, category, price, trend_score, success_score, risk])

print("\n📁 הנתונים נשמרו לקובץ: aicommerce_data.csv")
