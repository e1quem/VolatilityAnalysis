import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
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
tokenRequest = requests.get(f'https://gamma-api.polymarket.com/markets?slug={marketName}').json()
tokenList = json.loads(tokenRequest[0]['clobTokenIds'])
token = tokenList[0]

# API request for price history
res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')

# Dataframe creation
dfFull = pd.DataFrame(res.json()['history'])
dfFull['t'] = pd.to_datetime(dfFull['t'], unit='s')
dfFull['p'] = (dfFull['p']*100).round(1)
dfFull.set_index('t', inplace=True)

# User input
days = round(len(dfFull)/144)
user_interval = input(f"\033[1mNumber of days to analyze\033[0m ({days} day(s) available): ")
interval = float(user_interval)

if interval > days:
    user_interval = input(f"\033[1mToo many days. Please select a number equal or smaller than {days}: \033[0m")
    interval = float(user_interval)

if interval < 1:
    aggregate = '1H'
elif 1 <= interval < 3:
    aggregate = '1H'
elif 3 <= interval < 7:
    aggregate = '1H'
else:
    aggregate = '4H'

pointsFull = int(interval * 144)
dfFull = dfFull.tail(pointsFull)

# Resampling data in candles for Rogers-Satchell analysis
df_resampled = dfFull['p'].resample(aggregate).ohlc().ffill()

# Calculating returns and auto-scaling them
df_resampled['logReturn'] = np.log(df_resampled['close'] / df_resampled['close'].shift(1))
std_dev = df_resampled['logReturn'].std()
scale = 1 / std_dev
ScaledReturns = df_resampled['logReturn'].dropna() * scale

# Calculating Rogers-Satchell variance
df_resampled['rs_variance'] = np.log(df_resampled['high'] / df_resampled['close']) * np.log(df_resampled['high'] / df_resampled['open']) + \
                              np.log(df_resampled['low'] / df_resampled['close']) * np.log(df_resampled['low'] / df_resampled['open'])

# Replacing 0s and shifting for exog alignment (lagged)
epsilon = 1e-6
df_resampled['x_var'] = (df_resampled['rs_variance'] * scale**2).shift(1).replace(0, epsilon).fillna(epsilon)
x_exog = df_resampled['x_var'].loc[ScaledReturns.index]

# Defining models specs
specs = {
    'ARCH(1)': {'p': 1, 'o': 0, 'q': 0, 'exog': None},
    'GARCH(1,1)': {'p': 1, 'o': 0, 'q': 1, 'exog': None},
    'TARCH(1,1,1)': {'p': 1, 'o': 1, 'q': 1, 'exog': None},
    # Rogers-Satchell as an exogenous variable
    'GARCH(1,1) RS': {'p': 1, 'o': 0, 'q': 1, 'exog': x_exog},
    'TARCH(1,1,1) RS': {'p': 1, 'o': 1, 'q': 1, 'exog': x_exog}
}

bicResults = {}

# BIC comparison for current period
for name, prm in specs.items():
    try:
        model = arch_model(ScaledReturns, p=prm['p'], o=prm['o'], q=prm['q'], x=prm['exog'], dist='ged')
        fit_tmp = model.fit(update_freq=0, disp='off', show_warning=False)
        if fit_tmp.convergence_flag == 0:
            bicResults[name] = fit_tmp.bic
    except: continue

bicDF = pd.Series(bicResults, name='BIC').sort_values()
print(f"\n        \033[1m{aggregate} forecast")
print(" Model name         BIC\033[0m")
print(bicDF)

# Fitting best model
bestModelName = bicDF.index[0]
prm = specs[bestModelName]
bestModel = arch_model(ScaledReturns, p=prm['p'], o=prm['o'], q=prm['q'], x=prm['exog'], dist='ged')
fit = bestModel.fit(update_freq=0, disp='off')

# Forecast
last_x = df_resampled['x_var'].iloc[-1] if prm['exog'] is not None else None
forecast = fit.forecast(horizon=1, x=last_x)
forecastVolScaled = np.sqrt(forecast.variance.iloc[-1, 0])
forecastVol = (forecastVolScaled / scale) * np.sqrt(365)
print(f"Forecasted volatility: {forecastVol:.2f}%")


# Backtesting
trainingPoints = 40 
if len(ScaledReturns) <= trainingPoints + 10:
    print("Not enough data for significant backtesting.")
else:
    resultsBT = []
    print(f"\nBacktesting {len(ScaledReturns) - trainingPoints - 1} data points...")

    for i in range(trainingPoints, len(ScaledReturns) - 1):
        train_data = ScaledReturns.iloc[:i]
        best_bic = float('inf')
        current_best_name = None

        for name, prm in specs.items():
            curr_x = prm['exog'].iloc[:i] if prm['exog'] is not None else None
            try:
                m_tmp = arch_model(train_data, p=prm['p'], o=prm['o'], q=prm['q'], x=curr_x, dist='ged')
                res_tmp = m_tmp.fit(update_freq=0, disp='off', show_warning=False)
                if res_tmp.convergence_flag == 0 and res_tmp.bic < best_bic:
                    best_bic = res_tmp.bic
                    current_best_name = name
            except: continue

        if current_best_name:
            b_prm = specs[current_best_name]
            train_x = b_prm['exog'].iloc[:i] if b_prm['exog'] is not None else None
            
            try:
                model_final = arch_model(train_data, p=b_prm['p'], o=b_prm['o'], q=b_prm['q'], x=train_x, dist='ged')
                fit_bt = model_final.fit(update_freq=0, disp='off', show_warning=False)
                
                next_x = b_prm['exog'].iloc[i] if b_prm['exog'] is not None else None
                f_bt = fit_bt.forecast(horizon=1, x=next_x)
                
                sigma_raw = np.sqrt(f_bt.variance.iloc[-1, 0]) / scale
                current_price = df_resampled['close'].iloc[i]
                next_price = df_resampled['close'].iloc[i+1]
                
                lo = max(0, current_price * np.exp(-1.96 * sigma_raw))
                hi = min(100, current_price * np.exp(1.96 * sigma_raw))
                
                hit = lo <= next_price <= hi
                miss_type = None

                if not hit:
                    miss_type = 'up' if next_price > hi else 'down'
                    
                resultsBT.append({
                    't': df_resampled.index[i+1],
                    'price': next_price,
                    'lo': max(0, lo),
                    'hi': min(100, hi),
                    'hit': hit,
                    'model': current_best_name,
                    'miss_type': miss_type
                })
            except: continue

    # Results analysis
    dfBT = pd.DataFrame(resultsBT).set_index('t')

    dfFull = dfFull.sort_index()
    dfBT = dfBT.sort_index()
    limits = dfBT[['lo', 'hi']]

    final = pd.merge_asof(dfFull[['p']], limits, left_index=True, right_index=True)
    final.dropna(inplace=True)

    # Checking hits comparing the aggregated prediction to the 10m data
    final['hit_10m'] = (final['p'] >= final['lo']) & (final['p'] <= final['hi'])
    accuracy_10m = final['hit_10m'].mean() * 100
    hits = final['hit_10m'].sum()
    miss_up = (final['p'] > final['hi']).sum()
    miss_down = (final['p'] < final['lo']).sum()
    total_points = len(final)

    model_counts = dfBT['model'].value_counts()
    print("\n\033[1m Models used during backtest\033[0m")
    for m_name, count in model_counts.items():
        percentage = (count / len(dfBT)) * 100
        print(f"{m_name.ljust(12)}: {count} points ({percentage:.2f}%)")

    print(f"\n\033[1m        Results\033[0m")
    print(f"10m points: {total_points}")
    print(f"Hits      : {hits} ({hits/total_points*100:.2f}%)")
    print(f"Miss Up   : {miss_up} ({miss_up/total_points*100:.2f}%)")
    print(f"Miss Down : {miss_down} ({miss_down/total_points*100:.2f}%)")
    print(f"Accuracy  : {accuracy_10m:.2f}%")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(final.index, final['p'], color='black', label='Price (10m)', linewidth=1, zorder=3)
    plt.fill_between(final.index, final['lo'], final['hi'], color='orange', alpha=0.2, label='95% interval')

    miss_up = final[final['p'] > final['hi']]
    plt.scatter(miss_up.index, miss_up['p'], color='red', s=10, zorder=5, label='Miss up')

    miss_down = final[final['p'] < final['lo']]
    plt.scatter(miss_down.index, miss_down['p'], color='limegreen', s=10, zorder=5, label='Miss down')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.title(f'GARCH RS | {marketName[:30]}\n Accuracy: {accuracy_10m:.2f}% | Best model: {bestModelName}')

    plt.ylabel('Price (c)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()