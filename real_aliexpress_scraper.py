import requests
from bs4 import BeautifulSoup
import json
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def scrape_aliexpress(keyword="projector"):
    url = f"https://www.aliexpress.com/wholesale?SearchText={keyword}"
    r = requests.get(url, headers=HEADERS, timeout=20)

    soup = BeautifulSoup(r.text, "lxml")
    products = []

    items = soup.select("a.multi--container--1UZxxHY")[:20]

    for item in items:
        try:
            title = item.get("title")
            link = "https:" + item.get("href")

            img = item.select_one("img")
            image = img.get("src") or img.get("data-src")

            price = item.select_one(".multi--price-sale--U-S0jtj")
            price = price.text.strip() if price else "לא זמין"

            product = {
                "title": title,
                "link": link,
                "price": price,
                "image": image,
                "orders_now": None,   # מוגן ע"י AliExpress
                "category": keyword
            }

            products.append(product)
            time.sleep(0.4)

        except Exception as e:
            continue

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("✅ נשמרו מוצרים אמיתיים עם תמונות")
    print(f"✅ נמצאו {len(products)} מוצרים")

if __name__ == "__main__":
    scrape_aliexpress("projector")
