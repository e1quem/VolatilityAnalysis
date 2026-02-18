import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings 
import requests
import socket
import utils
import json

warnings.filterwarnings("ignore")
utils.ipv4()

# Obtaining market info
marketName = input("\033[1mMarket slug\033[0m (end of the Polymarket URL): ")

# Obtaining market token
tokenRequest = requests.get(f'https://gamma-api.polymarket.com/markets?slug={marketName}')
curl = tokenRequest.json()
tokens = curl[0]['clobTokenIds']
tokenList = json.loads(tokens)
token = tokenList[0]

# API request for price history
res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&fidelity=10&interval=max')

# Dataframe creation
df = pd.DataFrame(res.json()['history'])
df['t'] = pd.to_datetime(df['t'], unit='s')
df['p'] = (df['p']*100).round(1)
df.set_index('t', inplace=True)
df = df[~df.index.duplicated(keep='last')]
df = df.sort_index()

# User input for amount of days to analyze
days = round(len(df)/144)
user_interval = input(f"\033[1mNumber of days to analyze\033[0m ({days} day(s) available): ")
interval = float(user_interval)

if interval > days:
    user_interval = input(f"\033[1mToo many days. Please select a number equal or smaller than {days}:\033[0m ")
    interval = float(user_interval)

# Seting the ranges of the HAR-RV model
if interval < 1:
    hd, hw, hm = 6, 18, 36
elif 1 <= interval < 3:
    hd, hw, hm = 24, 72, 144
elif 3 <= interval < 7:
    hd, hw, hm = 24, 72, 144
else:
    hd, hw, hm = 24, 72, 144

pointsFull = int(interval * 144)
print(f"{len(df)} data points")

# Calculating returns
df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
df['Return2'] = df['logReturn'].dropna() ** 2

# Application of HAR-RV settings
df['RVD'] = df['Return2'].rolling(hd, min_periods=1).mean()
df['RVW'] = df['Return2'].rolling(hw, min_periods=1).mean()
df['RVM'] = df['Return2'].rolling(hm, min_periods=1).mean()
harDF = df[['RVD','RVM', 'RVW', 'Return2']].dropna(subset=['Return2'])

# Forecast horizons
horizons = {"10min": 1}

# Backtesting
start_index = hm
backtest_results = {}

for name, h in horizons.items():
    results = []
    end_range = len(harDF) - h
    
    print(f"Backtest: {name} horizon")

    if start_index >= end_range:
        print(f"Not enough data points for {name} horizon.")
        continue
    
    # Applying HAR-RV model
    for i in range(start_index, len(harDF) - h):
        train_data = harDF.iloc[:i]
        
        X_train = sm.add_constant(train_data[['RVD', 'RVW', 'RVM']])
        y_train = train_data['Return2'].shift(-h).dropna()
        X_train = X_train.loc[y_train.index]
        
        model_h = sm.OLS(y_train, X_train).fit()
        
        current_features = [1, train_data['RVD'].iloc[-1], train_data['RVW'].iloc[-1], train_data['RVM'].iloc[-1]]
        pred_R2_h = max(0, model_h.predict(current_features)[0])
        
        sigma_h = np.sqrt(pred_R2_h)
        per95 = sigma_h * 1.96
        
        priceNow = df['p'].loc[harDF.index[i]]
        priceFuture = df['p'].loc[harDF.index[i+h]]
        
        lo, hi = max(0, priceNow * (1 - per95)), min(100, priceNow * (1 + per95))

        hit = lo <= priceFuture <= hi
        miss_type = None

        if not hit:
            miss_type = 'up' if priceFuture > hi else 'down'
        
        results.append({
            't': harDF.index[i+h],
            'price': priceFuture,
            'lo': lo,
            'up': hi,
            'hit': hit,
            'miss_type': miss_type
        })
    
    if results: 
        backtest_results[name] = pd.DataFrame(results).set_index('t')
    else: 
        print(f"No results for {name}")

# Results
bt_df = backtest_results["10min"]
accuracy = bt_df['hit'].mean() * 100
print("\n\033[1mResults\033[0m")
print(f"{accuracy:.2f}% acc")

# Plot
plt.figure(figsize=(12, 6))

plt.plot(bt_df.index, bt_df['price'], color='black', label='Price', linewidth=1, zorder=3)

plt.fill_between(bt_df.index, bt_df['lo'], bt_df['up'], color='orange', alpha=0.2, label='95% interval', step='post')

miss_up = bt_df[bt_df['miss_type'] == 'up']
plt.scatter(miss_up.index, miss_up['price'], color='red', s=10, zorder=5, label='Miss up')

miss_down = bt_df[bt_df['miss_type'] == 'down']
plt.scatter(miss_down.index, miss_down['price'], color='limegreen', s=10, zorder=5, label='Miss down')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
plt.title(f'HAR-RV | {marketName:.50}\n Accuracy: {accuracy:.2f}%')

plt.ylabel('Price (c)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()