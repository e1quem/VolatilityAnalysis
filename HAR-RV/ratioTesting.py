import matplotlib.pyplot as plt
from datetime import datetime
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

# Obtaining market data
marketName = input("\033[1mMarket slug\033[0m: ")
tokenRequest = requests.get(f'https://gamma-api.polymarket.com/markets?slug={marketName}')
token = json.loads(tokenRequest.json()[0]['clobTokenIds'])[0]

res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&fidelity=10&interval=max')

# Dataframe creation
df = pd.DataFrame(res.json()['history'])
df['t'] = pd.to_datetime(df['t'], unit='s')
df['p'] = (df['p']*100).round(1)
df.set_index('t', inplace=True)
df = df[~df.index.duplicated(keep='last')].sort_index()

days = round(len(df)/144)
interval = float(input(f"\033[1mDays to analyze\033[0m ({days} available): "))

# Calculating returns
df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
df['Returns2'] = df['logReturn'] ** 2

# List of ratios to test
# HAR-RV is 1d, 1w, 1m (2:14:61)
# Here, 1 corresponds to a 10m interval
ratios = [
    # Classic
    (1, 7, 30),
    (2, 14, 61),
    (3, 21, 90),
    (4, 28, 122),
    (5, 35, 150),
    (6, 42, 183),
    (7, 49, 210),
    (8, 56, 244),
    (9, 63, 270),
    (10, 70, 305),
    (12, 84, 360),
    (24, 168, 720),

    # Fast paced
    (1, 3, 6),
    (2, 6, 12),
    (3, 6, 12),
    (6, 12, 24),
    (1, 2, 4),
    (2, 4, 8),
    (4, 8, 16),

    # Else
    (24, 48, 72),
    (24, 72, 144),
    (6, 24, 48),
    (6, 48, 144),
    (12, 144, 504),
    (144, 504, 1008),
    (8, 21, 55),
    (13, 34, 144),
]

all_results = []

print(f"\nTesting {len(ratios)} ratios on {len(df)} points")

for (hd, hw, hm) in ratios:
    # Calculating moving averages with each setting
    df['RVD'] = df['Returns2'].rolling(hd, min_periods=1).mean()
    df['RVW'] = df['Returns2'].rolling(hw, min_periods=1).mean()
    df['RVM'] = df['Returns2'].rolling(hm, min_periods=1).mean()
    
    harDF = df[['RVD', 'RVW', 'RVM', 'Returns2']].dropna(subset=['Returns2'])
    
    results = []
    h = 1 # 10min horizon
    start_index = hm
    
    if start_index >= len(harDF) - h:
        print(f"Skipping {hd}:{hw}:{hm} - Not enough data.")
        continue

    # Testing loop
    for i in range(start_index, len(harDF) - h):
        train_data = harDF.iloc[:i]
        
        X_train = sm.add_constant(train_data[['RVD', 'RVW', 'RVM']])
        y_train = train_data['Returns2'].shift(-h).dropna()
        X_train = X_train.loc[y_train.index]
        
        model = sm.OLS(y_train, X_train).fit()
        
        current_feat = [1, train_data['RVD'].iloc[-1], train_data['RVW'].iloc[-1], train_data['RVM'].iloc[-1]]
        pred_R2 = max(0, model.predict(current_feat)[0])
        
        sigma = np.sqrt(pred_R2)
        price_now = harDF.index[i]
        
        p_now = df['p'].loc[harDF.index[i]]
        p_fut = df['p'].loc[harDF.index[i+h]]
        
        lo, hi = max(0, p_now * (1 - sigma * 1.96)), min(100,p_now * (1 + sigma * 1.96))
        results.append(lo <= p_fut <= hi)

    if results:
        acc = (sum(results) / len(results)) * 100
        all_results.append((f"{hd}:{hw}:{hm}", acc))
        print(f"{hd}:{hw}:{hm}: {acc:.2f}%")

# Results
print("\033[1mRanking\033[0m")
all_results.sort(key=lambda x: x[1], reverse=True)
for ratio, accuracy in all_results:
    print(f"{ratio.ljust(12)} : {accuracy:.2f}%")