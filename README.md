# E-commerce Competitor Strategy Dashboard

## Project Overview
This project is a comprehensive **E-commerce Competitor Strategy Dashboard** designed to assist businesses in analyzing competitor data, predicting trends, and generating actionable strategies to optimize pricing, promotions, and customer satisfaction. The dashboard integrates data scraping, machine learning, and large language models (LLMs) to provide meaningful insights.

## Features

### 1. **Data Collection**
- Competitor pricing, discounts, and reviews are scraped daily from e-commerce platforms (e.g., Amazon) using BeautifulSoup.
- The data is stored in two CSV files:
  - `competitor_data.csv`: Contains product pricing, discounts, and dates.
  - `reviews.csv`: Contains customer reviews for each product.

### 2. **Sentiment Analysis**
- Customer reviews are analyzed for sentiment using the Hugging Face Transformers library.
- Sentiment results are visualized in a bar chart using Plotly.

### 3. **Predictive Modeling**
- A Random Forest Regressor model predicts discounts based on product prices.
- An ARIMA model forecasts future discounts for the next five days.

### 4. **Strategy Recommendations**
- The dashboard leverages OpenAI's API to generate actionable strategies based on:
  - Competitor data (current and predicted trends).
  - Customer sentiment analysis.

### 5. **Slack Integration**
- Generated strategy recommendations are sent to a Slack channel for immediate access by stakeholders.

### 6. **Interactive Dashboard**
- Built with Streamlit, the dashboard allows users to:
  - Select a product for analysis.
  - View competitor data, predicted discounts, and sentiment analysis.
  - Read detailed strategy recommendations.

---

## How It Works

### **Data Scraping**
- URLs for products are predefined and scraped using BeautifulSoup.
- Functions extract product titles, prices, discounts, and customer reviews.

### **Data Preprocessing**
- Product prices and discounts are cleaned and converted to numerical formats.
- Dates are parsed to ensure proper indexing for time-series forecasting.

### **Machine Learning Models**
- **Random Forest Regressor**: Predicts potential discounts based on price and historical discount data.
- **ARIMA**: Forecasts future discounts for a specified number of days.

### **Sentiment Analysis**
- Hugging Face's `pipeline()` is used to analyze customer sentiment (Positive, Negative, Neutral).
- Results are visualized as an interactive bar chart in the dashboard.

### **LLM Integration**
- OpenAI's API is utilized to generate strategy recommendations tailored to the selected product's data and sentiment trends.

---

## Files in the Repository

### 1. **Main Dashboard Code**
- **`app.py`**: Main Streamlit app integrating data loading, visualization, sentiment analysis, forecasting, and strategy generation.

### 2. **Scraper Code**
- **`scraper.py`**: Scrapes competitor product data and reviews from e-commerce platforms and saves them as CSV files.

### 3. **Data Files**
- **`competitor_data.csv`**: Contains scraped competitor data with fields:
  - `product_name`, `price`, `discount`, `date`
- **`reviews.csv`**: Contains customer reviews with fields:
  - `product_name`, `reviews`

### 4. **API Keys**
- **`API_KEY.py`** (Not shared in the repository):
  - Contains API keys for Groq API and Slack Webhook integration.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or above
- Libraries:
  - `pandas`, `numpy`, `streamlit`, `plotly`, `requests`, `beautifulsoup4`,`selenium`, `scikit-learn`, `statsmodels`, `transformers`
- API keys for:
  - OpenAI (Groq API)
  - Slack Webhook

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecommerce-strategy-dashboard.git
   cd ecommerce-strategy-dashboard
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your API keys to `API_KEY.py`:
   ```python
   GROQ_API_KEY = "your-groq-api-key"
   SLACK_WEBHOOK_API_KEY = "your-slack-webhook-key"
   ```
4. Run the scraper to collect competitor data:
   ```bash
   python scraper.py
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

---

## Outputs

### Dashboard
- **Competitor Analysis**: View competitor pricing, discounts, and trends.
- **Sentiment Analysis**: Interactive sentiment bar chart.
- **Discount Forecasting**: Predicted discounts for the next 5 days.
- **Strategy Recommendations**: AI-generated strategies for pricing, promotions, and customer satisfaction.

### Slack Notifications
- Automatically sends generated strategies to a specified Slack channel.

---

## Future Enhancements
- Expand data scraping to additional e-commerce platforms.
- Integrate more advanced NLP models for nuanced sentiment analysis.
- Incorporate real-time data updates and streaming.

---

## Screenshots

<div style="display: flex; flex-wrap: wrap;">
  <img src="./Screenshots/1.png" alt="Screenshot 1" width="400" style="margin: 10px;">
  <img src="./Screenshots/2.png" alt="Screenshot 2" width="400" style="margin: 10px;">
  <img src="./Screenshots/3.png" alt="Screenshot 3" width="400" style="margin: 10px;">
  <img src="./Screenshots/4.png" alt="Screenshot 4" width="400" style="margin: 10px;">
  <img src="./Screenshots/5.png" alt="Screenshot 5" width="400" style="margin: 10px;">
  <img src="./Screenshots/6.png" alt="Screenshot 6" width="400" style="margin: 10px;">
  <img src="./Screenshots/7.png" alt="Screenshot 7" width="400" style="margin: 10px;">
</div>


