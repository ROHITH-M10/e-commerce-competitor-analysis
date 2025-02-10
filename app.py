import pandas as pd  # For handling dataframes and CSV files
import plotly.express as px  # For visualizing data with interactive plots
import requests  # For making HTTP requests (e.g., sending Slack messages, API calls)
import streamlit as st  # For creating the interactive dashboard
# from openai import AzureOpenAI  # (Commented out) for using OpenAI models via Azure
# from sklearn.ensemble import RandomForestRegressor  # Machine learning model for price prediction
from sklearn.model_selection import train_test_split  # Splitting data into training and testing
from statsmodels.tsa.arima.model import ARIMA  # ARIMA model for time series forecasting
from transformers import pipeline  # For sentiment analysis using NLP models
from datetime import datetime  # Handling dates and timestamps
import json  # Handling JSON data


GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # Groq API Key
SLACK_WEBHOOK_API_KEY = st.secrets["SLACK_WEBHOOK_API_KEY"] # Slack Webhook url


def truncate_text(text, max_length=512):
    return text[:max_length] 

def load_competitor_data():
    # load competitor data from competitor_data.csv
    data = pd.read_csv("competitor_data.csv")
    return data

def load_reviews_data():
    # load reviews data from reviews_data.csv
    reviews = pd.read_csv("reviews.csv")
    return reviews


def analyze_sentiment(reviews):
    """Perform sentiment analysis on customer reviews using a pre-trained NLP model."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)

# def train_predictive_model(data):
#     data['discount'] = data['discount'].str.replace('%', '').astype(float)
#     data['price'] = data['price'].str.replace('$', '').astype(float)
#     data['predicted_discount'] = data['discount'] + (data['price'] * 0.05).round(2)

#     X = data[['price', 'discount']]
#     y = data['predicted_discount']
#     # print(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#         )
    
#     model = RandomForestRegressor(random_state=42)
#     model.fit(X_train, y_train)
#     return model

import numpy as np
import pandas as pd

def forecast_discounts_arima(data,future_days = 5):
    """
    Forcast future discounts using ARIMA model
    :param data: pandas DataFrame historical discount data with datetime index
    :param future_days: int number of days to forecast
    :return: pandas DataFrame with forecasted discount data
    """

    data = data.sort_index()
    # print(product_data.index)
    # Convert to numeric
    data['discount'] = pd.to_numeric(data['discount'], errors='coerce')
    # Drop rows with missing discount values
    data = data.dropna(subset=['discount'])

    discount_series = data['discount']

    # print(discount_series)
    # Check if index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(
                "Index could not be converted to datetime"
                ) from e

    # Train an ARIMA model (5,1,0 parameters indicate autoregressive, differencing, moving average)
    model = ARIMA(discount_series, order=(5,1,0))
    model_fit = model.fit()

    # Forecast future discounts
    forecast = model_fit.forecast(steps=future_days)
    # convet to int and round off
    forecast = np.round(forecast).astype(int)

    # Generate future dates
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    ).date

    # Create a DataFrame with forecasted data
    forecast_df = pd.DataFrame({'date': future_dates, 'predicted_discount': forecast})
    forecast_df.set_index('date', inplace=True)

    return forecast_df

def send_to_slack(data):
    # Send message to slack

    payload = {"text": data}
    response = requests.post(
        SLACK_WEBHOOK_API_KEY,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def generate_strategy_recommendation(product_name, competitor_data, predicted_discounts, sentiment):
    # generate strategy recommendation using an LLM model
    date = datetime.now()

    # prompt for the model
    prompt = f"""
    You are a highly skilled business strategist specilizing in e-commerce. 
    Based on the following details, 
    suggest actionable strategies to optimize pricing (data is available in rupees)
    promotions and customer satisfaction 
    for the selected product

    1. **Product Name**: {product_name}

    2. **Competitor Data** (including current prices, discounts for last 5 days and predicted discount for next 5 days):
    {competitor_data}
    **Predicted Discounts**: {predicted_discounts}

    3. **Sentiment Analysis**: {sentiment}

    4. **Today's Date**: {str(date)}


    ### Task:
    - Analyze the competitor data and identify key pricing trends.
    - Leverage the sentiment analysis to higlight areas where customer satisfaction can be improved.
    - Use the discount predictions to suggest how pricing strategies can be optimized over next 5 days.
    - Recommend promotional campaigns or marketing strategies that align with customer sentiment and competitive trends.
    - Ensure the strategies are actionable, realistic, and geared towards increasing customer satisfaction, driving sales and maximizing profits.

    Provide your recommendations in a structured format:
    1. **Pricing Strategy**
    2. **Promotional Campaign Ideas**
    3. **Customer Satisfaction Recommendations**
    
    """

    message = [{"role":"user","content":prompt}]

    data = {
        "messages": message,
        "model": "llama3-8b-8192",
        "temperature" : 0
    }

    headers = {"Context-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}

    # make API request to Groq
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data = json.dumps(data),
        headers=headers,
    )

    # print("API Response:", res)

    res = res.json()
    # print("Response:", res)
    
    # extract response from the API
    response = res['choices'][0]['message']['content']
    return response




#-------------- Streamlit App -----------------#

st.set_page_config(page_title="E-commerce Competitor Strategy Dashboard", layout="wide")

st.title("E-commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

# List of products to choose from
products = [
    "boAt Rockerz 255",
    "Oneplus Bullets Z2",
    "Realme Buds Wireless 3 Neo",
    "JBL Tune 215BT"
]

# Select a product to analyze
selected_product = st.sidebar.selectbox("Select a Product to analyze", products)

# Load data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter data for the selected product
product_data = competitor_data[competitor_data['product_name'] == selected_product]
product_reviews = reviews_data[reviews_data['product_name'] == selected_product]

print("++++++++")
print(product_data.shape)


# print("++++++++")
# print(product_data.tail())


st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")

# Display the last 5 rows of the competitor data
st.table(product_data.tail(5))

# Sentiment Analysis of Customer Reviews
if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x,512)
    )

    reviews = product_reviews["reviews"].tolist()
    sentiments = analyze_sentiment(reviews)

    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)

    # Plot the sentiment analysis results
    fig = px.bar(sentiment_df, x='label', title='Sentiment Analysis Results')
    st.plotly_chart(fig)

else:
    st.write("No reviews data available for this product.")

# print("++++++++")
# print(product_data)

# Convert 'date' to datetime using the correct format
product_data['date'] = pd.to_datetime(product_data['date'], format='%Y-%m-%d')
# print("++++++++")
# print(product_data)



# Drop rows with missing dates
product_data = product_data.dropna(subset=['date'])

# Set 'date' as the index
product_data.set_index("date", inplace=True)

# Sort the data by date
product_data = product_data.sort_index()


# Forecasting future discounts
product_data['discount'] = pd.to_numeric(product_data['discount'], errors='coerce')
product_data = product_data.dropna(subset=['discount'])

# Forecasting future discounts
product_data_with_predictions = forecast_discounts_arima(product_data)



st.subheader("Competitor Predicted Discounts")
# Display the last 5 rows of the competitor data with predicted discounts
st.table(product_data_with_predictions.tail(5))

# Use the data to train a predictive model
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data[['price', 'discount']][-5:],
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews data available"
    )

# Display the strategy recommendations
st.subheader("Strategy Recommendations")
st.write(recommendations)
# Send the recommendations to Slack
send_to_slack(recommendations)
