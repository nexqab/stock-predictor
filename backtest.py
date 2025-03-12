def backtest_stock(company_name, start_date="2023-01-01"):
    cerebro = bt.Cerebro()
    
    # ✅ Convert company name to ticker symbol
    symbol = search_ticker_symbol(company_name)

    if not symbol:
        print(f"❌ No ticker found for {company_name}")
        return

    print(f"📊 Downloading historical data for {symbol}...")

    # ✅ Ensure `yfinance` returns a valid DataFrame
    stock_data = yf.download(symbol, start=start_date, auto_adjust=False, progress=False)

    # ✅ Handle missing data
    if stock_data is None or stock_data.empty:
        print(f"❌ No historical data found for {company_name} ({symbol})")
        return

    # ✅ Ensure correct column names for Backtrader
    stock_data = stock_data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
    )

    # ✅ Ensure DataFrame index is in DateTime format
    stock_data.index = pd.to_datetime(stock_data.index)

    # ✅ DEBUG: Print first few rows for verification
    print(stock_data.head())

    data = bt.feeds.PandasData(dataname=stock_data)
    cerebro.adddata(data)

    cerebro.addstrategy(AIStockStrategy)
    cerebro.run()
    cerebro.plot()
