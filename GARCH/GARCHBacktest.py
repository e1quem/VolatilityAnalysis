# Apply GARCH volatility models to price history of individuals markets
# Trains the models then compares their BIC to forecast 95% confidence price intervals for the next period
# Defines hits and misses by comparing aggregated price to 10m price
# Graphic visualisation

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

# User input for amount of days to analyze
days = round(len(dfFull)/144)
userInterval = input(f"\033[1mNumber of days to analyze\033[0m ({days} day(s) available): ")
interval = float(userInterval)

if interval > days:
    userInterval = input(f"\033[1mToo many days. Please select a number equal or smaller than {days}:\033[0m ")
    interval = float(userInterval)
if interval < 1:
    aggregate = '10min' # Already in 10 minutes intervals
elif 1 <= interval < 3:
    aggregate = '20min'
elif 3 <= interval < 7:
    aggregate = '1H'
else:
    aggregate = '4H'

pointsFull = int(interval * 144)
dfFull = dfFull.tail(pointsFull)
df = dfFull.copy()

# Resampling the data: base data is in 10min intervals, we adapte the granularity according to the horizon needed
df = df['p'].resample(aggregate).last().ffill().to_frame()
points = {'10min': 144, '20min' : 72, '1H': 24, '4H': 6}
pointsToKeep = int(interval * points[aggregate])
df = df.tail(pointsToKeep)
print(f"Analyzing {userInterval} day(s) with a granularity of {aggregate} ({len(df)} data points).")

# Calculating log returns
df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
std = df['logReturn'].std()
scale = 1 / std
ScaledReturns = df['logReturn'].dropna() * scale # Autoscaling returns for arch stability


# Selecting the best volatility model via BIC comparison
specs = {
    'ARCH(1)': (1,0,0),
    'ARCH(2)': (2,0,0),
    'GARCH(1,1)': (1,0,1),
    'GARCH(2,1)': (2,0,1),
    'TARCH(1,1,1)': (1,1,1),
    'TARCH(2,2,1)': (2,2,1)}
bicResults = {}

for name, (p, o, q) in specs.items():
    modelSTD = arch_model(ScaledReturns, p=p, o=o, q=q, dist='ged')
    fitSTD = modelSTD.fit(update_freq=0, disp='off', show_warning=False) # We avoid wordy display of results with disp='off'.
    bicResults[name] = fitSTD.bic

bicDF = pd.Series(bicResults, name='BIC').sort_values()

print(f"\n        \033[1m{aggregate} forecast")
print(" Model name         BIC\033[0m")
print(bicDF)

# Fitting the best model
bestModelName = min(bicResults, key=bicResults.get)
p, o, q = specs[bestModelName]
bestModel = arch_model(ScaledReturns, p=p, o=o, q=q)
fit = bestModel.fit(update_freq=0, disp='off')
df['vol'] = (fit.conditional_volatility/scale) * np.sqrt(365)

# Volatility forecast
forecast = fit.forecast(horizon=1)
forecastVolScaled = np.sqrt(forecast.variance.iloc[-1, 0])
forecastVol = (forecastVolScaled / scale) * np.sqrt(365)

# Simple 95% confidence price forecast
per95 = (forecastVolScaled / scale)  * 1.96 
price = dfFull['p'].iloc[-1]

lo = max(0, price * (1 - per95))
hi = min(100, price * (1 + per95))

print(f"\nVolatility : {forecastVol:.2f}%")
print(f"95% interval: from {lo:.2f}c to {hi:.2f}c")


# Backtesting
# Defining the amount of points needed to fit the model
trainingPoints = 40 

if len(ScaledReturns) <= trainingPoints+10:
    print("Not enough data for significant backtesting.")
else:
    resultsBT = []

    print(f"\nBacktesting {len(ScaledReturns) - trainingPoints} data points...")

    for i in range(trainingPoints, len(ScaledReturns) - 1):
        train_data = ScaledReturns.iloc[:i]
        # For each step, selection of the best model using the BIC method
        best_bic = float('inf')
        best_params = (1, 0, 1) 
        
        # Fitting the models
        for name, (pb, ob, qb) in specs.items():
            try:
                m_tmp = arch_model(train_data, p=pb, o=ob, q=qb, dist='ged')
                res_tmp = m_tmp.fit(update_freq=0, disp='off', show_warning=False)

                # Compare the BIC only if the model is properly fitted
                if res_tmp.convergence_flag == 0:
                    if res_tmp.bic < best_bic:
                        best_bic = res_tmp.bic
                        best_params = (pb, ob, qb)
            except: continue
        
        # For each step, the best model is used for t+1 price boundaries forecasting
        try:
            p_opt, o_opt, q_opt = best_params
            fit_bt = arch_model(train_data, p=p_opt, o=o_opt, q=q_opt, dist='ged').fit(update_freq=0, disp='off', show_warning=False)
            
            # Forecast only if the model is properly fitted
            if fit_bt.convergence_flag != 0: continue
                
            # 95% confidence price forecast
            current_price = df['p'].iloc[i]
            next_price = df['p'].iloc[i+1]
            time_idx = df.index[i+1]

            sigma = np.sqrt(fit_bt.forecast(horizon=1).variance.iloc[-1, 0]) / scale

            lo = max(0, current_price * np.exp(-1.96 * sigma))
            hi = min (100, current_price * np.exp(1.96 * sigma))

            hit = lo <= next_price <= hi
            miss_type = None

            if not hit:
                miss_type = 'up' if next_price > hi else 'down'
            
            # Checking which model we were using
            name_lookup = {v: k for k, v in specs.items()}
            current_best_name = name_lookup.get(best_params, "Unknown")

            # Saving results
            resultsBT.append({
                't': time_idx,
                'price': next_price,
                'lo': lo,
                'hi': hi,
                'hit': hit,
                'model': current_best_name,
                'miss_type': miss_type
            })
        except: continue

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
plt.title(f'GARCH | {marketName:.50}\n Accuracy: {accuracy_10m:.2f}%')

plt.ylabel('Price (c)')
plt.xlabel('Time')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()