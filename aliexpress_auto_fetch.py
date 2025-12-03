import requests
from bs4 import BeautifulSoup
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def fetch_aliexpress_products(keyword="gadgets", limit=10):
    url = f"https://www.aliexpress.com/wholesale?SearchText={keyword.replace(' ', '+')}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    products = []

    for item in soup.select("a")[:200]:
        text = item.get_text(strip=True)
        if len(text) > 20 and len(products) < limit:
            products.append({
                "name": text[:60],
                "price": random.randint(20, 150),   # מחיר משוער (אפשר לשדרג)
                "orders_now": random.randint(100, 5000)  # הזמנות משוערות
            })

    return products
