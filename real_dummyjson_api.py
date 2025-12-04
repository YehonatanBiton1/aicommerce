import requests
import json

def fetch_real_products():
    url = "https://dummyjson.com/products?limit=30"
    response = requests.get(url)
    data = response.json()

    products = []

    for item in data.get("products", []):
        product = {
            "title": item.get("title"),
            "price": item.get("price"),
            "category": item.get("category"),
            "orders_now": item.get("stock"),
            "link": f"https://dummyjson.com/products/{item.get('id')}",
            "image": item.get("thumbnail")
        }

        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"✅ נשמרו {len(products)} מוצרים אמיתיים!")

if __name__ == "__main__":
    fetch_real_products()
