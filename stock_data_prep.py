import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Function to load Fear and Greed Index data from a local CSV file
def load_fear_greed_data(filepath='fear-greed-2011-2023.csv'):
    try:
        print(f"Loading Fear & Greed Index data from {filepath}...")
        fear_greed = pd.read_csv(filepath)
        print(f"CSV columns: {fear_greed.columns.tolist()}")
        if 'Date' in fear_greed.columns and 'Fear Greed' in fear_greed.columns:
            fear_greed = fear_greed.rename(columns={'Date': 'date', 'Fear Greed': 'fear_greed_value'})
        fear_greed['date'] = pd.to_datetime(fear_greed['date'])
        fear_greed = fear_greed.set_index('date')
        print(f"Successfully loaded Fear & Greed data with {len(fear_greed)} entries")
        return fear_greed
    except Exception as e:
        print(f"Error loading Fear & Greed data: {e}")
        return create_synthetic_sentiment()

# Function to create synthetic sentiment data as a fallback (hopefully not used)
def create_synthetic_sentiment():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    n = len(trading_days)
    changes = np.random.normal(0, 1, n)
    sentiment = 50 + np.cumsum(changes)
    for i in range(1, n):
        reversion = 0.05 * (50 - sentiment[i])
        sentiment[i] += reversion
    sentiment = np.clip(sentiment, 0, 100)
    sentiment_df = pd.DataFrame({'fear_greed_value': sentiment}, index=trading_days)
    print(f"Created synthetic market sentiment data with {len(sentiment_df)} entries")
    return sentiment_df

# Function to add technical indicators to stock data
def add_technical_indicators(df):
    data = df.copy()
    
    # Moving Averages (using short-horizon windows)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Moving Average Crossovers (only include the short-term ones)
    data['MA5_cross_MA20'] = (data['MA5'] > data['MA20']).astype(int)
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
    data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
    
    # MACD and Signal
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    data['MACD_signal_cross'] = ((data['MACD'] > data['MACD_signal']) &
                                 (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))).astype(int)
    
    # Bollinger Bands (using 20-day window)
    data['MA20_std'] = data['Close'].rolling(window=20).std()
    data['upper_band'] = data['MA20'] + (data['MA20_std'] * 2)
    data['lower_band'] = data['MA20'] - (data['MA20_std'] * 2)
    close_arr = data['Close'].to_numpy().flatten()
    lower_arr = data['lower_band'].to_numpy().flatten()
    upper_arr = data['upper_band'].to_numpy().flatten()
    diff_arr = upper_arr - lower_arr
    diff_arr[diff_arr == 0] = np.nan  # avoid division by zero
    bb_position = (close_arr - lower_arr) / diff_arr
    data['BB_position'] = pd.Series(bb_position, index=data.index)
    
    # Volatility measures (20-day and 50-day)
    ma20_std_arr = data['Close'].rolling(window=20).std().to_numpy().flatten()
    ma20_arr = data['MA20'].replace(0, np.nan).to_numpy().flatten()
    volatility_20d = (ma20_std_arr / ma20_arr) * 100
    data['volatility_20d'] = pd.Series(volatility_20d, index=data.index)
    
    ma50_std_arr = data['Close'].rolling(window=50).std().to_numpy().flatten()
    ma50_arr = data['MA50'].replace(0, np.nan).to_numpy().flatten()
    volatility_50d = (ma50_std_arr / ma50_arr) * 100
    data['volatility_50d'] = pd.Series(volatility_50d, index=data.index)
    
    # Price Rate of Change
    data['ROC_5'] = data['Close'].pct_change(periods=5) * 100
    data['ROC_10'] = data['Close'].pct_change(periods=10) * 100
    data['ROC_20'] = data['Close'].pct_change(periods=20) * 100
    
    # Volume indicators
    if 'Volume' in data.columns:
        data['volume_MA20'] = data['Volume'].rolling(window=20).mean()
        volume_ma20_arr = data['volume_MA20'].replace(0, np.nan).to_numpy().flatten()
        volume_arr = data['Volume'].to_numpy().flatten()
        volume_ratio = volume_arr / volume_ma20_arr
        data['volume_ratio'] = pd.Series(volume_ratio, index=data.index)
        price_diff = data['Close'].diff().to_numpy().flatten()
        direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        obv = (volume_arr * direction).cumsum()
        data['OBV'] = pd.Series(obv, index=data.index)
    
    # Price gaps and daily returns
    data['gap_up'] = (data['Open'] > data['Close'].shift(1)).astype(int)
    data['gap_down'] = (data['Open'] < data['Close'].shift(1)).astype(int)
    data['daily_return'] = data['Close'].pct_change() * 100
    
    # Target: predict next day's direction (1 if next day's Close > today's Close)
    data['target_direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data

# Main function to process SPY data and output a CSV file
def process_spy_data(fear_greed_file='fear-greed-2011-2023.csv'):
    ticker = 'SPY'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"Could not retrieve data for {ticker}")
        return None
    print(f"Original data shape: {stock_data.shape}")
    
    # Add technical indicators
    print("Adding technical indicators...")
    data = add_technical_indicators(stock_data)
    print(f"Data shape after adding indicators: {data.shape}")
    
    # Flatten multi-level columns if necessary (for yfinance data, e.g., ('Close','SPY') becomes 'Close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Reset index to convert the date index into a simple column
    data = data.reset_index()
    print("Columns after resetting index:", data.columns.tolist())
    
    # Rename the date column to 'date' if needed
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'date'})
    elif 'index' in data.columns:
        data = data.rename(columns={'index': 'date'})
    
    # Load Fear & Greed Index data (reset its index so that 'date' is a column)
    fear_greed = load_fear_greed_data(fear_greed_file)
    
    # Merge on the 'date' column
    if not fear_greed.empty:
        print("Merging Fear & Greed Index with stock data...")
        fg = fear_greed.reset_index()  # 'date' becomes a column here
        data = pd.merge(data, fg, how='left', on='date')
        data = data.set_index('date')
    
    # Download VIX index data (market volatility)
    print("Downloading VIX index data (market volatility)...")
    try:
        vix_data = yf.download('^VIX', start=start_date, end=end_date)
        # Flatten VIX columns if they are multi-level
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX'})
        vix_data = vix_data.reindex(data.index, method='ffill')
        data = data.join(vix_data)
        print("Successfully added VIX data")
    except Exception as e:
        print(f"Failed to add VIX data: {e}")
    
    # Data cleaning: drop any rows with NaN values
    data_cleaned = data.dropna()
    print(f"Data shape after dropping NaN values: {data_cleaned.shape}")
    
    # Save the processed data to a CSV file (overwriting any previous file)
    full_data_filename = f"{ticker}_full_processed_data.csv"
    data_cleaned.to_csv(full_data_filename)
    print(f"Saved full processed data to {full_data_filename}")
    
    return {'ticker': ticker, 'data': data_cleaned}

# Run processing and generate CSV output
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("PROCESSING SPY - S&P 500 ETF")
    print(f"{'='*50}\n")
    
    fear_greed_file = 'fear-greed-2011-2023.csv'
    result = process_spy_data(fear_greed_file)
    
    if result:
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE - CSV output generated")
        print(f"{'='*50}")
    else:
        print("Failed to process SPY data")
