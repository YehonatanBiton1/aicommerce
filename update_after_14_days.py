import pandas as pd

DATA_PATH = "aicommerce_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n--- מוצרים שממתינים לעדכון ---")
pending = df[df["orders_after_14_days"].isna()]
print(pending[["name","orders_now"]])

name = input("\nהכנס שם מוצר לעדכון: ")

row_index = df[df["name"] == name].index
if len(row_index) == 0:
    print("❌ מוצר לא נמצא")
    exit()

new_orders = int(input("כמות הזמנות חדשה (אחרי 14 יום): "))

old_orders = int(df.loc[row_index[0], "orders_now"])
diff = new_orders - old_orders

real_success = 1 if diff >= 200 else 0

df.loc[row_index[0], "orders_after_14_days"] = new_orders
df.loc[row_index[0], "real_success"] = real_success

df.to_csv(DATA_PATH, index=False, encoding="utf-8")

print("✅ עודכן בהצלחה!")
print("📈 שינוי בהזמנות:", diff)
print("🏆 הצלחה?" , "כן" if real_success == 1 else "לא")
