import requests
import json

EBAY_CLIENT_ID = "PUT_CLIENT_ID_HERE"  # כאן תכניס את המפתח כשתקבל

def fetch_from_ebay(query="projector"):
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"

    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }

    params = {
        "q": query,
        "limit": 20
    }

    r = requests.get(url, headers=headers, params=params)
    data = r.json()

    products = []

    for item in data.get("itemSummaries", []):
        product = {
            "title": item.get("title"),
            "price": item.get("price", {}).get("value"),
            "image": item.get("image", {}).get("imageUrl"),
            "link": item.get("itemWebUrl"),
            "orders_now": item.get("estimatedAvailableQuantity", 0),
            "category": query
        }
        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("✅ נטענו מוצרים אמיתיים מ-eBay")
