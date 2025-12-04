from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, json

def scrape_aliexpress(keyword="projector"):
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)

    url = f"https://www.aliexpress.com/wholesale?SearchText={keyword}"
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/item/']"))
        )
    except:
        print("❌ לא נמצאו מוצרים – כנראה חסימת בוט")
        driver.quit()
        return

    cards = driver.find_elements(By.CSS_SELECTOR, "a[href*='/item/']")

    print("🔎 נמצא מספר כרטיסים:", len(cards))

    products = []

    for card in cards[:20]:
        link = card.get_attribute("href")
        title = card.text.strip()

        if not title or not link:
            continue

        product = {
            "title": title,
            "link": link,
            "price": 0,
            "orders_now": 0,
            "category": keyword,
            "image": ""
        }

        products.append(product)

    driver.quit()

    with open("market_products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("✅ נשמרו בפועל:", len(products), "מוצרים")

if __name__ == "__main__":
    scrape_aliexpress("projector")
