import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aicommerce_data.csv")

print("\n=== טבלת נתוני AICommerce ===")
print(df)

print("\n--- סטטיסטיקה בסיסית ---")
print(df[["price", "trend_score", "success_score"]].describe())

print("\n--- פיזור קטגוריות ---")
print(df["category"].value_counts())

# ---------- תובנות ----------
avg_success = df["success_score"].mean()
best_category = df.groupby("category")["success_score"].mean().idxmax()
worst_category = df.groupby("category")["success_score"].mean().idxmin()
best_product = df.loc[df["success_score"].idxmax()]
worst_product = df.loc[df["success_score"].idxmin()]

print(f"\nממוצע Success Score כללי: {avg_success:.1f}")
print(f"הקטגוריה החזקה ביותר: {best_category}")
print(f"הקטגוריה החלשה ביותר: {worst_category}")
print(f"המוצר החזק ביותר: {best_product['name']} ({best_product['success_score']})")
print(f"המוצר החלש ביותר: {worst_product['name']} ({worst_product['success_score']})")

# ---------- גרף ----------
plt.scatter(df["price"], df["success_score"])
plt.xlabel("מחיר (₪)")
plt.ylabel("Success Score")
plt.title("גרף מחיר מול Success Score")
plt.grid(True)
plt.show()
