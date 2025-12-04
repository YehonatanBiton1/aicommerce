import requests
import json


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}


def scrape_mercadolibre(keyword="ergonomic chair", site_id="MLA"):
    url = f"https://api.mercadolibre.com/sites/{site_id}/search"

    params = {
        "q": keyword,
        "limit": 20
    }

    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)

    if resp.status_code != 200:
        print("❌ שגיאה מגובה MercadoLibre:", resp.status_code)
        print(resp.text)
        return

    data = resp.json()
    products = []

    for item in data.get("results", []):
        product = {
            "title": item.get("title", "").strip(),
            "price": float(item.get("price", 0)),
            "link": item.get("permalink", ""),
            "image": (item.get("thumbnail") or "").replace("http://", "https://"),
            "orders_now": item.get("sold_quantity", 0),
            "category": keyword
        }
        products.append(product)

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"✅ נשמרו {len(products)} מוצרים אמיתיים ל-market_products.json")


if __name__ == "__main__":
    scrape_mercadolibre("ergonomic chair")
