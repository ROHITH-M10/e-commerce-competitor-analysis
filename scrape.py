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

# Define the product links with corresponding names
links = {
    "boAt Rockerz 255" : "https://www.amazon.in/boAt-Rockerz-255-Pro-Earphones/dp/B08TV2P1N8",
    "Oneplus Bullets Z2": "https://www.amazon.in/Oneplus-Bluetooth-Wireless-Earphones-Bombastic/dp/B09TVVGXWS",
    "Realme Buds Wireless 3 Neo":"https://www.amazon.in/realme-Buds-Wireless-Bluetooth-Resistannt/dp/B0D3HT2S1M",
    "JBL Tune 215BT":"https://www.amazon.in/JBL-Playtime-Bluetooth-Earphones-Assistant/dp/B08FB2LNSZ"
}

# Function to extract all reviews from a product page
def get_all_reviews(driver):
    reviews = []
    try:
        # Find all review elements
        review_divs = driver.find_elements(By.CSS_SELECTOR, "div.a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
        for review_div in review_divs:
            # Extract text from each review span
            review_span = review_div.find_element(By.TAG_NAME, "span")
            if review_span:
                review = review_span.text.strip()
                reviews.append(review)
    except Exception as e:
        print(f"Error occurred while extracting reviews: {e}")
    return reviews

# Function to scrape product data including price, discount, and reviews
def scrape_product_data(link):
    options = Options()  # Create an instance of the Options class to configure Chrome WebDriver
    options.add_argument("--headless")  # Run Chrome in headless mode (without a GUI)
    options.add_argument("--no-sandbox")  # Disable the sandbox for security restrictions (useful in containerized environments)
    options.add_argument("--disable-dev-shm-usage")  # Prevents issues with shared memory in Docker or limited memory environments
    options.add_argument("--disable-gpu")  # Disables GPU hardware acceleration (useful in headless mode to avoid rendering issues)
    options.add_argument("--lang=en")  # Set the browser's default language to English
    options.add_argument("--window-size=1920,1080")  # Set the browser window size to 1920x1080 for consistency in rendering


    # Initialize Chrome WebDriver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.set_window_size(1920, 1080)
    driver.get(link)  # Open the product page
    
    product_data = {"reviews": []}  # Initialize dictionary to store product data
    wait = WebDriverWait(driver, 10) # Initialize WebDriverWait with a timeout of 10 seconds
    time.sleep(5)  # Wait for the page to load
    
    # Retry loop to handle exceptions
    retry = 0
    while retry < 3:
        try:
            driver.save_screenshot("screenshot.png")  # Take a screenshot for debugging
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen"))) # Wait for the product name to load

            # Extract reviews
            reviews = get_all_reviews(driver) # Get all reviews by calling the get_all_reviews function
            product_data["reviews"] = reviews  # Store reviews

            break  # Exit retry loop on success
        except Exception as e:
            print("Retrying due to error:", e)
            retry += 1
            driver.get(link)  # Reload the page
            time.sleep(5)

        driver.save_screenshot("screenshot.png")  # Capture final screenshot
    
    try:
        # Extract selling price
        price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]',
        )
        product_data["selling_price"] = int("".join(price_elem.text.strip().split(",")))
    except:
        product_data["selling_price"] = 0  # Default to 0 if not found

    try:
        # Extract original price
        original_price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]',
        )
        product_data["original_price"] = int("".join(original_price_elem.text.strip().split(","))) # Remove commas and convert to integer
    except:
        product_data["original_price"] = 0  # Default to 0 if not found
    
    try:
        # Extract discount percentage or rating
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
            product_data["discount"] = full_rating_text[1:-1] # Remove percentage sign 
    except:
        product_data["discount"] = 0  # Default to 0 if not found

    product_data["date"] = time.strftime("%Y-%m-%d")  # Store current date
    driver.quit()  # Close WebDriver
    return product_data

# Main scraping loop
for product_name, link in links.items(): # Iterate over each product link
    product_data = scrape_product_data(link)
    try:
        # Load existing reviews if available
        reviews = json.loads(pd.read_csv("reviews.csv").to_json(orient="records"))
    except pd.errors.EmptyDataError:
        reviews = []
    
    try:
        # Load existing price data if available
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

    # Convert to DataFrame for processing
    price_df = pd.DataFrame(price)
    # Convert date column to datetime format
    price_df['date'] = pd.to_datetime(price_df['date'], format='mixed', dayfirst=True)
    # Sort by product name and date
    price_df.sort_values(by=['product_name', 'date'], inplace=True)
    # Drop duplicate records
    price_df.drop_duplicates(inplace=True)  # Remove duplicate records
    # Drop null values
    price_df.dropna(inplace=True)  # Remove null values

    # Print number of duplicate records
    print("=== Duplicates ===", price_df.duplicated().sum())

    # Print number of reviews scraped
    print(f"Scraped {len(product_data['reviews'])} reviews for {product_name}")
    
    # Append new reviews
    for review in product_data["reviews"]:
        reviews.append({"product_name": product_name, "reviews": review})

    # Convert to DataFrame for processing
    reviews_df = pd.DataFrame(reviews)
    # Sort by product name
    reviews_df.sort_values(by=["product_name"], inplace=True)
    # Drop duplicate reviews
    reviews_df.drop_duplicates(inplace=True)  # Remove duplicate reviews
    # Drop null values
    reviews_df.dropna(inplace=True)  # Remove null values

    # Save data to CSV files
    reviews_df.to_csv("reviews.csv", index=False)
    price_df.to_csv("competitor_data.csv", index=False)

print("Scraping complete. Data saved to CSV files.")
