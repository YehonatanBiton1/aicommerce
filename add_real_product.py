import csv
from datetime import datetime

DATA_PATH = "aicommerce_data.csv"

email = input("אימייל: ")
name = input("שם מוצר מ-AliExpress: ")
category = input("קטגוריה: ")
price = float(input("מחיר ב-₪: "))
trend_score = float(input("Trend (0–100): "))
orders_now = int(input("כמות הזמנות נוכחית (Orders Now): "))

created_at = datetime.now().isoformat(timespec="seconds")

with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # אם הקובץ ריק – כותבים כותרת
    if f.tell() == 0:
        writer.writerow([
            "email","name","category","price","trend_score",
            "orders_now","orders_after_14_days","real_success","created_at"
        ])

    writer.writerow([
        email,
        name,
        category,
        price,
        trend_score,
        orders_now,
        "",   # orders_after_14_days (יתמלא מאוחר יותר)
        "",   # real_success (יתמלא אוטומטית)
        created_at
    ])

print("✅ המוצר נשמר לבדיקה עתידית")
