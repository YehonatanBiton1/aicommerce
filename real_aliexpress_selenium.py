from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import json
import time

def scrape_aliexpress(keyword="projector"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = f"https://www.aliexpress.com/wholesale?SearchText={keyword}"
    driver.get(url)
    time.sleep(7)

    products = []

    items = driver.find_elements(By.CSS_SELECTOR, "a.search-card-item")

    for item in items[:20]:
        try:
            title = item.get_attribute("title")
            link = item.get_attribute("href")

            img = item.find_element(By.TAG_NAME, "img")
            image = img.get_attribute("src")

            price_elem = item.find_element(By.CLASS_NAME, "manhattan--price-sale--1CCSZfK")
            price = price_elem.text

            product = {
                "title": title,
                "link": link,
                "price": price,
                "image": image,
                "orders_now": None,
                "category": keyword
            }

            products.append(product)

        except:
            continue

    driver.quit()

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"✅ נשמרו {len(products)} מוצרים אמיתיים עם תמונות")

if __name__ == "__main__":
    scrape_aliexpress("projector")
