import json
import time
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Define the product links
links = {
    "boAt Rockerz 255" : "https://www.amazon.in/boAt-Rockerz-255-Pro-Earphones/dp/B08TV2P1N8",
    "Oneplus Bullets Z2": "https://www.amazon.in/Oneplus-Bluetooth-Wireless-Earphones-Bombastic/dp/B09TVVGXWS",
    "Realme Buds Wireless 3 Neo":"https://www.amazon.in/realme-Buds-Wireless-Bluetooth-Resistannt/dp/B0D3HT2S1M",
    "JBL Tune 215BT":"https://www.amazon.in/JBL-Playtime-Bluetooth-Earphones-Assistant/dp/B08FB2LNSZ"
}

def get_all_reviews(driver):
    reviews = []
    try:
        review_divs = driver.find_elements(By.CSS_SELECTOR, "div.a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
        for review_div in review_divs:
            review_span = review_div.find_element(By.TAG_NAME, "span")
            if review_span:
                review = review_span.text.strip()
                reviews.append(review)
    except Exception as e:
        print(f"Error occurred while extracting reviews: {e}")
    return reviews

def scrape_product_data(link):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.set_window_size(1920, 1080)
    driver.get(link)
    
    product_data = {"reviews": []}  # Initialize reviews list
    wait = WebDriverWait(driver, 10)
    time.sleep(5)
    
    retry = 0
    while retry < 3:
        try:
            driver.save_screenshot("screenshot.png")
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))

            # Now we scrape reviews
            reviews = get_all_reviews(driver)
            product_data["reviews"] = reviews  # Store reviews

            break
        except Exception as e:
            print("Retrying due to error:", e)
            retry += 1
            driver.get(link)
            time.sleep(5)

    driver.save_screenshot("screenshot.png")
    
    try:
        price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]',
        )
        product_data["selling_price"] = int("".join(price_elem.text.strip().split(",")))
    except:
        product_data["selling_price"] = 0

    try:
        original_price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]',
        )
        product_data["original_price"] = int("".join(original_price_elem.text.strip().split(",")))
    except:
        product_data["original_price"] = 0
    
    try:
        discount = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]',
        )
        full_rating_text = discount.get_attribute("innerHTML").strip()
        if "out of 5 stars" in full_rating_text.lower():
            product_data["rating"] = (
                full_rating_text.lower().split(" out of")[0].strip()
            )
        else:
            product_data["discount"] = full_rating_text[1:-1]
    except:
        product_data["discount"] = 0

    product_data["date"] = time.strftime("%Y-%m-%d")
    driver.quit()
    return product_data

# Main scraping loop
for product_name, link in links.items():
    product_data = scrape_product_data(link)
    try:
        reviews = json.loads(pd.read_csv("reviews.csv").to_json(orient="records"))
    except pd.errors.EmptyDataError:
        reviews = []
    
    try:
        price = json.loads(pd.read_csv("competitor_data.csv").to_json(orient="records"))
    except pd.errors.EmptyDataError:
        price = []

    # Append new price data
    price.append({
        "product_name": product_name,
        "price": product_data["selling_price"],
        "discount": product_data["discount"],
        "date": datetime.now().strftime("%Y-%m-%d"),
    })

    price_df = pd.DataFrame(price)
    price_df['date'] = pd.to_datetime(price_df['date'], format='mixed', dayfirst=True)
    price_df.sort_values(by=['product_name', 'date'], inplace=True)

    # length of reviews
    print(f"Scraped {len(product_data['reviews'])} reviews for {product_name}")
    
    # Append new reviews
    for review in product_data["reviews"]:
        reviews.append({"product_name": product_name, "reviews": review})

    reviews_df = pd.DataFrame(reviews)
    # sort by product name
    reviews_df.sort_values(by=["product_name"], inplace=True)

    reviews_df.to_csv("reviews.csv", index=False)
    price_df.to_csv("competitor_data.csv", index=False)

print("Scraping complete. Data saved to CSV files.")
