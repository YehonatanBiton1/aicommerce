import requests
import json

API_KEY = ""
API_HOST = "walmart-api4.p.rapidapi.com"

def fetch_walmart_products(query="chair"):
    url = "https://walmart-api4.p.rapidapi.com/search"

    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST
    }

    params = {
        "query": query,
        "page": "1"
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    products = []

    for item in data.get("items", []):
        product = {
            "title": item.get("name"),
            "price": item.get("price", {}).get("price", 0),
            "category": item.get("categoryPath", "general"),
            "orders_now": 0,
            "link": item.get("canonicalUrl"),
            "image": item.get("image")
        }

        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"✅ נשמרו {len(products)} מוצרים אמיתיים מ-Walmart!")


if __name__ == "__main__":
    fetch_walmart_products("gaming chair")
