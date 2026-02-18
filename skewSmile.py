# Uses GARCH volatility models to plot observed volatility according to price in order to check for skew or smile

import matplotlib.pyplot as plt
from scipy.stats import skew
from arch import arch_model
import pandas as pd
import numpy as np
import warnings 
import requests
import utils
import socket
import json
import sys

warnings.filterwarnings("ignore")

utils.ipv4()
markets, volume = utils.getMarkets()
markets, typeName = utils.marketType(markets)

allData = []

# Loop to measure volatility using GARCH models
for idx, m_info in enumerate(markets):
    market_name = m_info['name']
    token = m_info['token']

    try:
        # API request for price history
        res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')

        # Dataframe creation
        dfHist = pd.DataFrame(res.json()['history'])
        dfHist['t'] = pd.to_datetime(dfHist['t'], unit='s')
        dfHist['p'] = (dfHist['p'] * 100).round(1)
        dfHist.set_index('t', inplace=True)

        # Checking the history of data available and adjusting parameters accordingly
        duration = (dfHist.index[-1] - dfHist.index[0]).total_seconds() / 86400
        if duration < 1:
            aggregate = '10min'
        elif 1 <= duration < 3:
            aggregate = '20min'
        elif 3 <= duration < 7:
            aggregate = '1H'
        else:
            aggregate = '4H'

        df = dfHist['p'].resample(aggregate).last().ffill().to_frame()

        # Calculating returns
        df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
        std = df['logReturn'].std()
        scale = 1 / std
        ScaledReturns = df['logReturn'].dropna() * scale # Autoscaling for arch stability

        trainingPoints = 40 
        if len(ScaledReturns) <= trainingPoints + 5:
            continue

        print(f"\033[1m[{idx+1}/{len(markets)}]\033[0m  {m_info['name'][:30]}...", end="\r")

        # Simple ARCH(1) model.
        # We'll explore more complexe models later on.
        m_tmp = arch_model(ScaledReturns, p=1, o=0, q=0, dist='ged')
        res_tmp = m_tmp.fit(update_freq=0, disp='off', show_warning=False)

        periods_per_year = (365 * 24 * 60) / int(aggregate.replace('min','').replace('H','00').replace('400','240').replace('100','60'))
        vol_local = (res_tmp.conditional_volatility / scale) * np.sqrt(periods_per_year)

        tempDF = pd.DataFrame({
            'price': df['p'].iloc[1:],
            'vol': vol_local
        }).dropna()
        
        allData.append(tempDF)


    except Exception as e:
        continue

# Ploting volatility smile/skew
if allData:
    df = pd.concat(allData)
    
    # Filter
    df = df[(df['vol'] < 5)]
    
    price_skew = skew(df['price'])
    print(f"\nPrice Skewness: {price_skew:.4f}")

    plt.figure(figsize=(12, 7))
    plt.scatter(df['price'], df['vol'] * 100, alpha=0.1, s=20, label='Volatility')

    # Using polyfit to simply fit the skew/smile
    z = np.polyfit(df['price'], df['vol'] * 100, 3)
    p = np.poly1d(z)
    x_range = np.linspace(df['price'].min(), df['price'].max(), 100)

    plt.plot(x_range, p(x_range), 'r', linewidth=2, label=f'Skew={price_skew:.2f})')

    plt.title(f'Volatility Skew: {len(markets)} {typeName} Markets ({volume}M+ Volume)')
    plt.xlabel('Price (c)')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, df['vol'].max() * 100)
    plt.show()
else:
    print("Not enough data.")