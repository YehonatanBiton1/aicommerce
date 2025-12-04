import requests
import json

def scrape_dummyjson():
    url = "https://dummyjson.com/products?limit=30"
    r = requests.get(url, timeout=20)

    data = r.json()
    products = []

    for item in data["products"]:
        product = {
            "title": item["title"],
            "price": float(item["price"]),
            "link": f"https://dummyjson.com/products/{item['id']}",
            "image": item["thumbnail"],
            "orders_now": int(item["rating"] * 100),
            "category": item["category"]
        }

        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"✅ נשמרו {len(products)} מוצרים אמיתיים + תמונות + קישורים!")

if __name__ == "__main__":
    scrape_dummyjson()
