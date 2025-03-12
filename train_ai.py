import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # To save/load AI models

def fetch_stock_data(symbol):
    """Fetch 6 months of stock data and calculate indicators."""
    print(f"📡 Fetching stock data for {symbol}...")

    try:
        stock_data = yf.download(symbol, period="6mo", interval="1d")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

    if stock_data.empty:
        print(f"❌ No stock data found for {symbol}. Please check the ticker.")
        return None

    stock_data.reset_index(inplace=True)

    # ✅ Fix column naming issue
    stock_data["SMA_50"] = stock_data["Close"].rolling(window=50, min_periods=1).mean()
    stock_data["SMA_200"] = stock_data["Close"].rolling(window=200, min_periods=1).mean()

    # ✅ RSI Calculation
    delta = stock_data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    stock_data["RSI"] = 100 - (100 / (1 + rs))

    stock_data.dropna(inplace=True)

    print(f"✅ Stock data successfully fetched! {len(stock_data)} rows available.")
    return stock_data


def train_ai(symbol):
    """Train an AI model to predict stock movement."""
    print(f"📊 Training AI model for {symbol}...")

    data = fetch_stock_data(symbol)
    if data is None or data.empty:
        print("❌ Error: No valid stock data found. Skipping training.")
        return

    # ✅ Creating Target column
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

    features = ["SMA_50", "SMA_200", "RSI"]

    # ✅ Check if required features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"❌ Missing required features: {missing_features}. Skipping training...")
        return

    X = data[features]
    y = data["Target"]

    # ✅ Require at least 50 data points for training
    if len(X) < 50:
        print(f"❌ Not enough data points for training. Found only {len(X)} samples.")
        return

    print(f"📊 Training Data Sample:\n{X.tail()}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, "ai_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print(f"✅ AI model trained and saved as `ai_model.pkl`!")


if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g., TSLA, AAPL, MSFT): ").strip().upper()
    train_ai(stock)
