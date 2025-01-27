from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

def get_title(soup):
    try:
        title = soup.find("span", attrs={"id": "productTitle"})
        title_value = title.text  # content inside span tag
        title_string = title_value.strip()
    except AttributeError:
        title_string = ""
    return title_string

def get_price(soup):
    try:
        price = soup.find("span", attrs={"class": "a-price-whole"}).text.strip()[:-1]
        price = price.replace(",", "")  # remove the comma from the price
    except:
        price = ""
    return price

def get_discount(soup):
    try:
        discount = soup.find(
            "span", attrs={"class": "a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage"}
        ).text.strip()[1:-1]
    except:
        discount = ""
    return discount

def get_rating(soup):
    try:
        rating = soup.find("span", attrs={"class": "a-icon-alt"}).text[:3]
    except:
        rating = ""
    return rating

def get_review(soup):
    try:
        review_div = soup.find("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content")
        if review_div:
            review_span = review_div.find("span")
            if review_span:
                review = review_span.get_text(strip=True)
                return review
    except:
        return ""

HEADERS = {
    'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US, en;q=0.5'
}

URLS = {
    "boAt Rockerz 255": "https://www.amazon.in/boAt-Rockerz-255-Pro-Earphones/dp/B08TV2P1N8",
    "Oneplus Bullets Z2": "https://www.amazon.in/Oneplus-Bluetooth-Wireless-Earphones-Bombastic/dp/B09TVVGXWS",
    "Realme Buds Wireless 3 Neo": "https://www.amazon.in/realme-Buds-Wireless-Bluetooth-Resistannt/dp/B0D3HT2S1M",
    "JBL Tune 215BT": "https://www.amazon.in/JBL-Playtime-Bluetooth-Earphones-Assistant/dp/B08FB2LNSZ"
}

# Scrape Price, Discount, Rating, Review for each product
competitor_data_today = pd.DataFrame(columns=["Product_name", "Price", "Date", "Discount"])
reviews_today = pd.DataFrame(columns=["Product_name", "Review"])

for product, url in URLS.items():
    competitor_data = {"Product_name": [], "Price": [], "Discount": [], "Date": time.strftime("%Y-%m-%d")}
    reviews_data = {"Product_name": [], "Review": []}

    page = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(page.content, "html.parser")

    # Extract data
    title = get_title(soup)
    price = get_price(soup)
    discount = get_discount(soup)
    review = get_review(soup)

    # Store competitor data
    competitor_data["Product_name"].append(product)
    competitor_data["Price"].append(price)
    competitor_data["Discount"].append(discount)
    competitor_data_today = pd.concat([competitor_data_today, pd.DataFrame(competitor_data)])

    # Store review data
    reviews_data["Product_name"].append(product)
    reviews_data["Review"].append(review)
    reviews_today = pd.concat([reviews_today, pd.DataFrame(reviews_data)])

# Save to CSV
today = time.strftime("%Y-%m-%d")
competitor_data_today.to_csv(f"competitor_data_{today}.csv", index=False)
reviews_today.to_csv(f"reviews_data_{today}.csv", index=False)
