# Plots distribution of 10m or periodically sampled log returns of markets.

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import warnings
import requests
import socket
import utils
import json
import sys

warnings.filterwarnings("ignore")

utils.ipv4()
markets, volume = utils.getMarkets()
markets, typeName = utils.marketType(markets)

all_returns = []

# Obtaining logReturns
for idx, m_info in enumerate(markets):
    try:
        # Obtaining price history (max fidelity is 10m, max history is 1 month)
        token = m_info['token']
        res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')
        
        df_hist = pd.DataFrame(res.json()['history'])
        if df_hist.empty: continue
        df_hist['t'] = pd.to_datetime(df_hist['t'], unit='s')
        df_hist.set_index('t', inplace=True)
        df_hist['p'] = df_hist['p'] * 100

        # Optional: periodical resample of returns
        #df_hourly = df_hist['p'].resample('1H').last().ffill()
        df_hourly = df_hist['p']

        log_returns = np.log(df_hourly / df_hourly.shift(1)).dropna()
        
        # Filtering out 0% logReturns, otherwise the majority of returns are aggregated at 0
        log_returns = log_returns[np.isfinite(log_returns)]
        log_returns = log_returns[log_returns != 0]
        
        all_returns.extend(log_returns.tolist())
        print(f"\033[1m[{idx+1}/{len(markets)}]\033[0m  Data collected for: {m_info['name'][:30]}...", end="\r")
        
    except Exception as e:
        continue

# Summary and analysis
skewness = stats.skew(all_returns)
kurtosis = stats.kurtosis(all_returns)
jb_stat, p_value = stats.jarque_bera(all_returns)

print("\n\033[1mAnalysis\033[0m")
print(f"Data points      : {len(all_returns)}")
print(f"Average logReturn: {np.mean(all_returns):.6f}")
print(f"StD              : {np.std(all_returns):.6f}")
print(f"Skewness         : {skewness:.4f}")
print(f"Excess Kurtosis  : {kurtosis:.4f}")
print(f"Jarque-Bera      : {jb_stat:.2f}")
print(f"p-value          : {p_value:.4e}")


# Plot
plt.figure(figsize=(12, 7))

lower_bound = np.percentile(all_returns, 0.5)
upper_bound = np.percentile(all_returns, 99.5)
custom_bins = np.arange(lower_bound, upper_bound, 0.002)

plt.hist(all_returns, bins=custom_bins, alpha=0.7, density=True, label='Global log-returns')

# Normal distribution for visual comparision.
mu = np.mean(all_returns)
sigma = np.std(all_returns)
x = np.linspace(lower_bound, upper_bound, 1000)
p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
plt.plot(x, p, 'r', linewidth=1.5, label=f'Normal distribution')

plt.title(f'Statistical Distribution of 10m Log-Returns: {len(markets)} {typeName} Markets ({volume}M+ Volume)', fontsize=12)
plt.ylabel('Density (%)')
plt.xlabel('Log-Return value')

plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.xlim(lower_bound, upper_bound)
plt.show()