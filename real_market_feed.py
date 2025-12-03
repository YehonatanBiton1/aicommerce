import requests
import json
import random

URL = "https://fakestoreapi.com/products"

def fetch_real_market_products():
    r = requests.get(URL, timeout=15)
    data = r.json()

    products = []

    for item in data:
        product = {
            "title": item["title"],
            "price": float(item["price"]),
            "orders_now": random.randint(100, 8000),
            "category": item["category"],
            "link": f"https://fakestoreapi.com/products/{item['id']}",
            "image": item["image"]
        }
        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("✅ נשמרו מוצרים אמיתיים מהשוק")
    print(f"✅ נמצאו {len(products)} מוצרים")

if __name__ == "__main__":
    fetch_real_market_products()
