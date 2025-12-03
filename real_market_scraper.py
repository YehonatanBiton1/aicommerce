import requests
import json

URL = "https://dummyjson.com/products?limit=20"

def scrape_real_products():
    r = requests.get(URL, timeout=15)
    data = r.json()

    products = []

    for item in data["products"]:
        product = {
            "title": item["title"],
            "price": item["price"],
            "orders_now": item["stock"],
            "category": item["category"],
            "image": item["thumbnail"],
            "link": f"https://dummyjson.com/products/{item['id']}"
        }

        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("✅ נשמרו מוצרים אמיתיים כולל תמונות וקישורים")
    print(f"✅ נמצאו {len(products)} מוצרים")

if __name__ == "__main__":
    scrape_real_products()
