from flask import Flask, render_template, request
import os
import openai
import yfinance as yf
import finnhub
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import joblib
from datetime import datetime

# ✅ Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

app = Flask(__name__)

# ✅ Load AI Model
def load_trained_ai_model():
    """Loads the pre-trained AI model and scaler."""
    try:
        model = joblib.load("ai_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading AI model: {e}")
        return None, None

# ✅ Get stock prognosis
def get_prognosis(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="5d")

    if hist.empty:
        return "Error: No stock data available."

    latest_close = hist["Close"].iloc[-1]
    avg_close = hist["Close"].mean()
    std_dev = hist["Close"].std()

    # ✅ Use AI Model
    model, scaler = load_trained_ai_model()
    if model and scaler:
        input_data = scaler.transform([[latest_close, avg_close, std_dev]])
        ai_prediction = model.predict(input_data)[0]
        forecast = "Up" if ai_prediction == 1 else "Down"
    else:
        forecast = "Unavailable"

    return forecast

@app.route("/", methods=["GET", "POST"])
def index():
    prognosis = None
    error = None

    if request.method == "POST":
        company = request.form.get("symbol")
        ticker_symbol = company.strip().upper()  # Assume valid ticker

        if ticker_symbol:
            prognosis_text = get_prognosis(ticker_symbol)

            if prognosis_text == "Error: No stock data available.":
                error = f"Could not retrieve data for {company}. Try again later."
            else:
                prognosis = {
                    "symbol": ticker_symbol,
                    "forecast": prognosis_text,
                    "timeframe": "1 Day"
                }
        else:
            error = f"Couldn't find a ticker symbol for '{company}'. Please verify spelling or try another name."

    return render_template("index.html", prognosis=prognosis, error=error)

if __name__ == "__main__":
    app.run(debug=True)
