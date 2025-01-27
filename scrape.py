from bs4 import BeautifulSoup
import requests
import pandas as pd
import schedule
import time

def get_title(soup):
  try:
    title = soup.find("span",attrs={"id":"productTitle"})
    title_value = title.text # content inside span tag
    title_string = title_value.strip()

  except AttributeError:
    title_string = ""
  
  return title_string

def get_price(soup):
    try:
        price = soup.find("span", attrs={"class": "a-price-whole"}).text.strip()[:-1]
        # remove the comma from the price
        price = price.replace(",", "")
    except:
        price = ""
    return price


def get_discount(soup):
    try:
        discount = soup.find("span", attrs={"class": "a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage"}).text.strip()[1:-1]
    except:
        discount = ""
    return discount


def get_rating(soup):
  try:
        rating = soup.find("span", attrs={"class": "a-icon-alt"}).text[:3]
  except:
        rating = ""
  return rating


def get_all_reviews(soup):
    reviews = []
    try:
        review_divs = soup.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content")
        for review_div in review_divs:
            review_span = review_div.find("span")
            if review_span:
                review = review_span.get_text(strip=True)
                reviews.append(review)
    except Exception as e:
        print(f"Error occurred: {e}")
    return reviews



HEADERS = ({'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36','Accept-Language': 'en-US, en;q=0.5'})


URLS = {
    "boAt Rockerz 255" : "https://www.amazon.in/boAt-Rockerz-255-Pro-Earphones/dp/B08TV2P1N8",
    "Oneplus Bullets Z2": "https://www.amazon.in/Oneplus-Bluetooth-Wireless-Earphones-Bombastic/dp/B09TVVGXWS/ref=sr_1_3?crid=1AD2BORMPFJHA&dib=eyJ2IjoiMSJ9.TCdYft94Omg-JxhEPSoOK2Y1bkzd6rS6K4SeCeQxH6p4Y9H8OcZ7iiZn7LK2LoClTDcKFcV5h3vCzYzLBd0VsFgdWJiPtOMuIzmHjOnNPZNxPbKA0sSxX1hjONJBqaEKzcFGKE0hmaFXmbr0aQ4igAuxXgXGTggmD0CY5IdJNIriYxcdTvEi55HTyBJg6O4Jz9wWEpG6N6TFh-R0tHdqW3fEMnWuDI8ldew88aJnEeY.gj-XLLo5c1Q5ukdSAiIzaDoFpVQBU-N3xbQOirwPxGg&dib_tag=se&keywords=bluetooth%2Bearphones%2Bwired&qid=1737911857&sprefix=bluetooth%2Bearphones%2Bwire%2Caps%2C270&sr=8-3&th=1",
    "Realme Buds Wireless 3 Neo":"https://www.amazon.in/realme-Buds-Wireless-Bluetooth-Resistannt/dp/B0D3HT2S1M/ref=sr_1_10?crid=1AD2BORMPFJHA&dib=eyJ2IjoiMSJ9.TCdYft94Omg-JxhEPSoOK2Y1bkzd6rS6K4SeCeQxH6p4Y9H8OcZ7iiZn7LK2LoClTDcKFcV5h3vCzYzLBd0VsFgdWJiPtOMuIzmHjOnNPZNxPbKA0sSxX1hjONJBqaEKzcFGKE0hmaFXmbr0aQ4igAuxXgXGTggmD0CY5IdJNIriYxcdTvEi55HTyBJg6O4Jz9wWEpG6N6TFh-R0tHdqW3fEMnWuDI8ldew88aJnEeY.gj-XLLo5c1Q5ukdSAiIzaDoFpVQBU-N3xbQOirwPxGg&dib_tag=se&keywords=bluetooth%2Bearphones%2Bwired&qid=1737911857&sprefix=bluetooth%2Bearphones%2Bwire%2Caps%2C270&sr=8-10&th=1",
    "JBL Tune 215BT":"https://www.amazon.in/JBL-Playtime-Bluetooth-Earphones-Assistant/dp/B08FB2LNSZ/ref=sr_1_18?crid=1AD2BORMPFJHA&dib=eyJ2IjoiMSJ9.TCdYft94Omg-JxhEPSoOK2Y1bkzd6rS6K4SeCeQxH6p4Y9H8OcZ7iiZn7LK2LoClTDcKFcV5h3vCzYzLBd0VsFgdWJiPtOMuIzmHjOnNPZNxPbKA0sSxX1hjONJBqaEKzcFGKE0hmaFXmbr0aQ4igAuxXgXGTggmD0CY5IdJNIriYxcdTvEi55HTyBJg6O4Jz9wWEpG6N6TFh-R0tHdqW3fEMnWuDI8ldew88aJnEeY.gj-XLLo5c1Q5ukdSAiIzaDoFpVQBU-N3xbQOirwPxGg&dib_tag=se&keywords=bluetooth%2Bearphones%2Bwired&qid=1737911857&sprefix=bluetooth%2Bearphones%2Bwire%2Caps%2C270&sr=8-18&th=1"

}

# Dataframe to store the scraped data - Product_name, Price, Discount, Date 
competitor_data_today = pd.DataFrame(columns=["Product_name", "Price", "Date"])


# Create review dataframe
# reviews_today = product_name, review
reviews_today = pd.DataFrame(columns=["Product_name", "Review"])


for product, url in URLS.items():
    # Create a dictionary to store the data
    competitor_data = {"Product_name": [], "Price": [], "Discount": [], "Date": time.strftime("%Y-%m-%d")}
    reviews_data = {"Product_name": [], "Review": []}
    
    # Get the page content
    page = requests.get(url, headers=HEADERS)
    
    # Create a BeautifulSoup object
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Get the title
    title = get_title(soup)
    
    # Get the price
    price = get_price(soup)
    
    # Get the discount
    discount = get_discount(soup)
    
    # Get all reviews
    all_reviews = get_all_reviews(soup)  # Use the updated function
    
    # Store the product data in the dictionary
    competitor_data["Product_name"].append(product)
    competitor_data["Price"].append(price)
    competitor_data["Discount"].append(discount)
    
    # Add the product data to the dataframe
    competitor_data_today = pd.concat([competitor_data_today, pd.DataFrame(competitor_data)])
    
    # Store all review data in the dictionary
    for review in all_reviews:
        reviews_data["Product_name"].append(product)
        reviews_data["Review"].append(review)
    
    # Add the review data to the dataframe
    reviews_today = pd.concat([reviews_today, pd.DataFrame(reviews_data)])


# Save the data to a CSV file with today's date
today = time.strftime("%Y-%m-%d")
competitor_data_today.to_csv(f"competitor_data_{today}.csv", index=False)


# Save the review data to a CSV file with today's date
reviews_today.to_csv(f"reviews_data_{today}.csv", index=False)