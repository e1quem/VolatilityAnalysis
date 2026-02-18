import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from arch import arch_model
import utils
import pandas as pd
import numpy as np
import warnings 
import requests
import socket
import json
import math
import sys

warnings.filterwarnings("ignore")

utils.ipv4()
markets, volume = utils.getMarkets()
markets, typeName = utils.marketType(markets)

summary_report = []
results = {}

for idx, m_info in enumerate(markets):
    market_name = m_info['name']
    token = m_info['token']

    try:
        # API request for price history
        res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')

        # Dataframe creation
        dfFull = pd.DataFrame(res.json()['history'])
        dfFull['t'] = pd.to_datetime(dfFull['t'], unit='s')
        dfFull['p'] = (dfFull['p'] * 100).round(1)
        dfFull.set_index('t', inplace=True)

        # Checking the history of data available and adjusting parameters accordingly
        duration = (dfFull.index[-1] - dfFull.index[0]).total_seconds() / 86400
        if duration < 1:
            aggregate = '10min' # Already in 10 minutes intervals
            dist = 'normal' # To fit the model more easily on shorter period of time
        elif 1 <= duration < 3:
            dist = 't'
        elif 3 <= duration < 7:
            aggregate = '1H'
            dist = 't'
        else:
            aggregate = '4H'
            dist = 't'

        df = dfFull['p'].resample(aggregate).last().ffill().to_frame()

        # Calculating log returns
        df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
        std_dev = df['logReturn'].std()
        scale = 1 / std_dev
        ScaledReturns = df['logReturn'].dropna() * scale # Autoscaling for arch stability

        # Backtesting
        # Defining the amount of points needed to fit the model
        trainingPoints = 40 
        specs = {
            'ARCH(1)': (1,0,0),
            'GARCH(1,1)': (1,0,1),
            'TARCH(1,1,1)': (1,1,1) # Removed GARCH(2,2), ARCH(2) and TARCH(2,2,1): they rarely had the lowest BIC on these markets.
        }
           
        if len(ScaledReturns) <= trainingPoints+10:
            continue

        start_index = trainingPoints
        resultsBT = []
        hits = 0
        totalTests = len(ScaledReturns)- start_index - 1

        print(f"\n\033[1m[{idx + 1}/{len(markets)}]\033[0m \033[3m{market_name} \033[0m")
        

        for i in range(start_index, len(ScaledReturns) - 1):
            train_data = ScaledReturns.iloc[:i]
            # For each step, selection of the best model using the BIC method
            best_bic = float('inf')
            best_params = (1, 0, 1) 
        
            # Fitting the models
            for name, (pb, ob, qb) in specs.items():
                try:
                    m_tmp = arch_model(train_data, p=pb, o=ob, q=qb, dist=dist)
                    res_tmp = m_tmp.fit(update_freq=0, disp='off', show_warning=False)

                    # Comparing the BIC only if the model is properly fitted
                    if res_tmp.convergence_flag == 0:
                        if res_tmp.bic < best_bic:
                            best_bic = res_tmp.bic
                            best_params = (pb, ob, qb)
                except: continue
        
            # For each step, the best model is used for t+1 price boundaries forecasting
            try:
                p_opt, o_opt, q_opt = best_params
                fit_bt = arch_model(train_data, p=p_opt, o=o_opt, q=q_opt, dist=dist).fit(update_freq=0, disp='off', show_warning=False)
                
                # Forecast only if the model is properly fitted
                if fit_bt.convergence_flag != 0: continue
                
                # Boundaries relying on the model
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

                resultsBT.append({
                    't': time_idx,
                    'price': next_price,
                    'lo': lo,
                    'hi': hi,
                    'hit': hit,
                    'model': name_lookup.get(best_params, "Unknown"),
                    'miss_type': miss_type
                })
            except: continue

        if resultsBT:
            dfBT = pd.DataFrame(resultsBT).set_index('t')

        if totalTests > 0:
            
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

            results[market_name] = final

            model_usage = pd.DataFrame(resultsBT)['model'].value_counts().idxmax() if resultsBT else "N/A"
            
            summary_report.append({
                'Market': market_name[:40], 
                '10m points': total_points,
                'Hits': hits,
                'Miss up': miss_up,
                'Miss down': miss_down,
                'Accuracy': round(accuracy_10m, 2),
                'Top Model': model_usage
            })
            print(f"      {accuracy_10m:.2f}% accuracy")
        else:
            print(f"Not enough data points to train the model.")

    except Exception as e:
        print(f"Error on {market_name}: {e}")
        continue

report_df = pd.DataFrame(summary_report)
if not report_df.empty:
    print(report_df.to_string(index=False))
else:
    print("No markets were successfully analyzed.")

# Plotting results
max_rows = 3
cols = 4
per_page = max_rows * cols
n_markets = len(results)
pages = math.ceil(n_markets / per_page)

items = list(results.items())

for page in range(pages):
    start = page * per_page
    end = start + per_page
    subset = items[start:end]

    rows = math.ceil(len(subset) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()

    for i, (m_name, final) in enumerate(subset):
        ax = axes[i]
        acc = (final['hit_10m'].sum() / len(final)) * 100

        ax.plot(final.index, final['p'], color='black', linewidth=1, label='Price', zorder=3)
        ax.fill_between(final.index, final['lo'], final['hi'], color='orange', alpha=0.2, label='95% interval')

        miss_up = final[final['p'] > final['hi']]
        ax.scatter(miss_up.index, miss_up['p'], color='red', s=10, zorder=5, label='Miss up')

        miss_down = final[final['p'] < final['lo']]
        ax.scatter(miss_down.index, miss_down['p'], color='limegreen', s=10, zorder=5, label='Miss down')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.set_title(f"{m_name[:40]}...\nAcc: {acc:.1f}%", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(alpha=0.2)
        #ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()