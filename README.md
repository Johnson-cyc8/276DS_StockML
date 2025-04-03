# 276DS_StockML

Using fear and greed data from https://github.com/whit3rabbit/fear-greed-data?tab=readme-ov-file


This CSV file contains processed SPY data with a variety of technical indicators computed from historical price and volume data. The indicators are derived from data obtained from Yahoo Finance and external sources (such as the Fear & Greed Index and VIX). The file is intended for training models to predict the stock's moving direction.

The CSV uses the date as its index and includes the following columns:

Close:
The daily closing price of SPY.

High:
The daily highest price of SPY.

Low:
The daily lowest price of SPY.

Open:
The daily opening price of SPY.

Volume:
The number of shares traded on that day.

MA5:
The 5-day simple moving average of the closing price, calculated as the average of the last 5 closing prices.

MA20:
The 20-day simple moving average of the closing price.

MA50:
The 50-day simple moving average of the closing price.

MA5_cross_MA20:
A binary indicator (1 or 0) that is 1 when MA5 is above MA20, suggesting a short-term bullish trend.

RSI:
The Relative Strength Index computed over a 14-day window. It measures momentum and ranges between 0 and 100.

RSI_oversold:
A binary indicator that is 1 if the RSI is below 30 (considered oversold).

RSI_overbought:
A binary indicator that is 1 if the RSI is above 70 (considered overbought).

EMA12:
The 12-day exponential moving average of the closing price.

EMA26:
The 26-day exponential moving average of the closing price.

MACD:
The Moving Average Convergence Divergence, computed as EMA12 minus EMA26.

MACD_signal:
The 9-day exponential moving average of the MACD, used as a signal line.

MACD_histogram:
The difference between the MACD and its signal line, showing the gap between them.

MACD_signal_cross:
A binary indicator that signals when the MACD crosses above its signal line.

MA20_std:
The 20-day rolling standard deviation of the closing price, used in Bollinger Bands.

upper_band:
The upper Bollinger Band, calculated as MA20 + (2 × MA20_std).

lower_band:
The lower Bollinger Band, calculated as MA20 – (2 × MA20_std).

BB_position:
The Bollinger Bands position (often called %B), computed as (Close – lower_band) / (upper_band – lower_band). It indicates where the price lies within the bands.

volatility_20d:
A measure of volatility computed as (20-day rolling standard deviation / MA20) × 100 (expressed as a percentage).

volatility_50d:
A measure of volatility computed as (50-day rolling standard deviation / MA50) × 100 (expressed as a percentage).

ROC_5:
The 5-day Rate of Change, calculated as the percentage change in the closing price over 5 days.

ROC_10:
The 10-day Rate of Change, calculated similarly over 10 days.

ROC_20:
The 20-day Rate of Change.

volume_MA20:
The 20-day simple moving average of the trading volume.

volume_ratio:
The ratio of the current day's volume to the 20-day moving average of volume, indicating unusual trading activity.

OBV:
On-Balance Volume, a cumulative indicator that combines volume with price movement direction to gauge buying and selling pressure.

gap_up:
A binary indicator set to 1 if the current day's opening price is higher than the previous day's closing price.

gap_down:
A binary indicator set to 1 if the current day's opening price is lower than the previous day's closing price.

daily_return:
The daily percentage return, calculated as the percentage change from the previous day’s close to the current day’s close.

target_direction:
The target variable: a binary indicator that is 1 if the next day’s closing price is higher than the current day’s closing price (0 otherwise).

fear_greed_value:
The Fear & Greed Index value for the day, an external measure of market sentiment. (The data is reindexed to match trading days by forward-filling.)

VIX:
The closing value of the VIX index, which measures market volatility.

Notes
Date Alignment:
The Fear & Greed Index is recorded on calendar days, but it is forward-filled to match the trading days of SPY. This means that weekends and holidays are effectively removed from the merged data.

Rolling Calculations:
Some indicators (e.g., moving averages, volatility, ROC) require a minimum number of historical data points. As a result, the first several rows of data are dropped, which is why the number of rows in the processed CSV is lower than the total number of trading days in the original dataset.

Usage for Weekly Prediction:
For predicting the weekly moving direction, you would use the values (computed up to Monday) as your input features to forecast the stock's direction over the week.


## License

[MIT License](LICENSE)
