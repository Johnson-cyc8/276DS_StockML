# 276DS_StockML

# Stock Price Direction Prediction Project

This project prepares data for a machine learning model that predicts the weekly direction of stock prices based on technical indicators and market sentiment data. The model predicts whether Friday's closing price will be higher or lower than Monday's opening price.

## Features

- **Technical Indicators**: EMA, Bollinger Bands, MACD, VWAP, ADX, RSI, and more
- **Market Sentiment Indicators**: Fear and Greed Index, VIX, Put/Call Ratio, Market Breadth, and Market Momentum
- **Weekly Data Structure**: Aggregates daily data into weekly features and targets
- **Data Visualization**: Includes visualizations of the prepared data
- **Mac Optimized**: Setup instructions for macOS users

## Setup Instructions

### Creating a Conda Environment

```bash
# Create a new environment named "stockml" with Python 3.10
conda create -n stockml python=3.10 -y

# Activate the environment
conda activate stockml

# Core data science packages from conda-forge (optimized for Mac)
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn -y

# Install additional packages
conda install -c conda-forge pandas-datareader -y
pip install yfinance ta requests

# Install ipykernel to make the environment available in Jupyter
conda install -c conda-forge ipykernel -y
python -m ipykernel install --user --name=stockml
```

### VSCode Setup

1. Open VSCode in your project directory:
```bash
mkdir -p ~/projects/stock-prediction
cd ~/projects/stock-prediction
code .
```

2. Select the "stockml" Python interpreter:
   - Press `Cmd+Shift+P` to open the command palette
   - Type "Python: Select Interpreter" and select it
   - Choose the "stockml" environment from the list

3. Create a launch configuration (`.vscode/launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {"PYTHONPATH": "${workspaceFolder}"},
            "python": "${command:python.interpreterPath}"
        }
    ]
}
```

## Data Sources

- **Stock Price Data**: Yahoo Finance (via yfinance)
- **Fear and Greed Index**: alternative.me API
- **VIX (Volatility Index)**: Yahoo Finance
- **Put/Call Ratio**: FRED Economic Data or proxy calculation
- **Market Breadth and Momentum**: Calculated from SPY ETF data

## Usage

Run the main script to prepare and visualize the data:

```bash
python stock_data_prep.py
```

This will:
1. Download stock price data and calculate technical indicators
2. Fetch market sentiment indicators from various sources
3. Create weekly features and calculate the target variable
4. Generate visualizations showing relationships in the data
5. Save the prepared dataset to a CSV file

## Customization

To analyze a different stock, modify the ticker variable:

```python
ticker = "AAPL"  # Change to your desired stock
```

To adjust the time period, modify the start and end dates:

```python
start_date = "2020-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')
```

## Output Files

- `{ticker}_weekly_data.csv`: Prepared dataset with all features and targets
- `{ticker}_data_visualization.png`: Visual summary of the prepared data

## Next Steps

After running the data preparation script, the resulting dataset is ready for model training. Recommended models include:

- XGBoost
- LightGBM
- Random Forest
- Neural Networks with LSTM layers

## License

[MIT License](LICENSE)
