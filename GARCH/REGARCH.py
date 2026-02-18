from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import warnings 
import requests
import socket
import json
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

def force_ipv4():
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        responses = old_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo

force_ipv4()

# --- Data Acquisition ---
marketName = 'will-there-be-no-change-in-fed-interest-rates-after-the-march-2026-meeting' 
tokenRequest = requests.get(f'https://gamma-api.polymarket.com/markets?slug={marketName}').json()
token = json.loads(tokenRequest[0]['clobTokenIds'])[0]
res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')

dfFull = pd.DataFrame(res.json()['history'])
dfFull['t'] = pd.to_datetime(dfFull['t'], unit='s')
dfFull['p'] = (dfFull['p']*100).round(1)
dfFull.set_index('t', inplace=True)

days_avail = round(len(dfFull)/144)
interval_input = float(input(f"\033[1mDays to analyze ({days_avail} available): \033[0m"))
aggregate = '4H' if interval_input >= 7 else '1H'

dfFull = dfFull.tail(int(interval_input * 144))
df_resampled = dfFull['p'].resample(aggregate).ohlc().ffill()
returns = np.log(df_resampled['close'] / df_resampled['close'].shift(1)).dropna()

# --- REGARCH Core Function ---
def regarch_log_likelihood(params, data_returns):
    # a1: s1->s2 speed, a2: s2->c2 speed, c2: target log-vol, b1: vol-of-vol1, b2: vol-of-vol2
    a1, a2, c2, b1, b2 = params
    if any(p <= 0 for p in [a1, a2, b1, b2]): return 1e15
    
    log_lik = 0
    ls1, ls2 = c2, c2 # Initial states
    
    for r in data_returns:
        sig = np.exp(ls1)
        # Log-vraisemblance Loi Normale
        log_lik += -0.5 * (np.log(2 * np.pi * sig**2) + (r**2 / sig**2))
        # Evolution déterministe (Euler) pour l'estimation
        ls1 += a1 * (ls2 - ls1)
        ls2 += a2 * (c2 - ls2)
        
    return -log_lik

# --- Rolling Backtest ---
training_points = 40
results_bt = []

if len(returns) <= training_points + 5:
    print("Not enough data points for backtesting.")
else:
    print(f"\n\033[3mStarting REGARCH Rolling Backtest ({len(returns)-training_points} iterations)...\033[0m")
    
    # Initial Guess [a1, a2, c2, b1, b2]
    current_params = [0.4, 0.1, np.log(returns.std()), 0.05, 0.01]

    for i in range(training_points, len(returns)):
        train_set = returns.iloc[i-training_points : i].values
        
        # Optimization (MLE)
        res_opt = minimize(regarch_log_likelihood, current_params, args=(train_set,), 
                           method='L-BFGS-B', bounds=[(1e-4, 2), (1e-4, 2), (-10, 2), (1e-4, 1), (1e-4, 1)])
        
        if res_opt.success:
            p = res_opt.x
            current_params = p # Hot-start pour l'itération suivante
            
            # Forecast T+1
            ls1, ls2 = p[2], p[2]
            for r in train_set:
                ls1 += p[0] * (ls2 - ls1)
                ls2 += p[1] * (p[2] - ls2)
            
            # Final predicted sigma for T+1
            sigma_pred = np.exp(ls1)
            
            current_price = df_resampled['close'].iloc[i] # Prix à T
            next_price = df_resampled['close'].iloc[i+1]    # Prix à T+1
            
            # Bornes 95%
            upper = current_price * np.exp(1.96 * sigma_pred)
            lower = current_price * np.exp(-1.96 * sigma_pred)
            
            results_bt.append({
                't': df_resampled.index[i+1],
                'price': next_price,
                'lower': lower,
                'upper': upper,
                'hit': lower <= next_price <= upper,
                'sigma': sigma_pred
            })
        
        if i % 10 == 0: print(f"Progress: {i}/{len(returns)}")

# --- Analysis & Plot ---
bt_df = pd.DataFrame(results_bt).set_index('t')
accuracy = bt_df['hit'].mean() * 100

print(f"\n\033[1mREGARCH Backtest Results\033[0m")
print(f"Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(14, 7))
plt.plot(bt_df.index, bt_df['price'], color='black', label='Price', linewidth=1.5, zorder=3)
plt.fill_between(bt_df.index, bt_df['lower'], bt_df['upper'], color='purple', alpha=0.2, label='REGARCH 95% CI')
plt.plot(bt_df.index, bt_df['lower'], color='purple', linestyle='--', alpha=0.3, linewidth=0.8)
plt.plot(bt_df.index, bt_df['upper'], color='purple', linestyle='--', alpha=0.3, linewidth=0.8)

# Highlight misses
misses = bt_df[~bt_df['hit']]
plt.scatter(misses.index, misses['price'], color='red', s=30, label='Misses', zorder=4)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
plt.title(f'REGARCH Rolling Backtest | Market: {marketName}\nAccuracy: {accuracy:.2f}%')
plt.legend(loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()