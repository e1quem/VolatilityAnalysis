from statsmodels.stats.stattools import durbin_watson
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
import math

warnings.filterwarnings("ignore")
utils.ipv4()
markets, volume = utils.getMarkets()
markets, typeName = utils.marketType(markets)

summary_report = []
all_market_results = {}

for idx, m_info in enumerate(markets):
    market_name = m_info['name']
    token = m_info['token']

    try:
        # API request for price history
        res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')

        # Dataframe creation
        df = pd.DataFrame(res.json()['history'])
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df['p'] = (df['p'] * 100).round(1)
        df.set_index('t', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        # Checking the history of data available and adjusting parameters accordingly
        duration = (df.index[-1] - df.index[0]).total_seconds() / 86400

        if duration < 1:
            hd, hw, hm = 6, 18, 36
        elif 1 <= duration < 3:
            hd, hw, hm = 24, 72, 144
        else:
            hd, hw, hm = 24, 72, 144

        pointsFull = int(duration * 144)

        # Calculating returns
        df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
        df['returns2'] = df['logReturn'].dropna() ** 2

        # Application of HAR-RV settings
        df['RVD'] = df['returns2'].rolling(hd, min_periods=1).mean()
        df['RVW'] = df['returns2'].rolling(hw, min_periods=1).mean()
        df['RVM'] = df['returns2'].rolling(hm, min_periods=1).mean()

        harDF = df[['RVD','RVM', 'RVW', 'returns2']].dropna(subset=['returns2'])

        start_index = hm
        resultsBT = []
        h = 1

        if len(harDF) <= start_index + 5:
            continue

        print(f"\033[1m[{idx + 1}/{len(markets)}]\033[0m \033[3m{market_name[:50]}...\033[0m")

        # Backtest loop
        for i in range(start_index, len(harDF) - h):
            train_data = harDF.iloc[:i]
            
            X_train = sm.add_constant(train_data[['RVD', 'RVW', 'RVM']])
            y_train = train_data['returns2'].shift(-h).dropna()
            X_train = X_train.loc[y_train.index]
            
            model = sm.OLS(y_train, X_train).fit()
            
            current_features = [1, train_data['RVD'].iloc[-1], train_data['RVW'].iloc[-1], train_data['RVM'].iloc[-1]]
            pred = max(0, model.predict(current_features)[0])
            RV = y_train.iloc[-1]

            # Residual
            residuals = pred - RV
            dw = durbin_watson(model.resid)
            
            sigma = np.sqrt(pred)
            p_now = df['p'].loc[harDF.index[i]]
            p_fut = df['p'].loc[harDF.index[i+h]]
            
            # 95% interval confidence
            lo, hi = max(0, p_now * (1 - sigma * 1.96)), min(100, p_now * (1 + sigma * 1.96))
            hit = lo <= p_fut <= hi
            miss_type = None

            if not hit:
                miss_type = 'up' if p_fut > hi else 'down'
            
            resultsBT.append({
                't': harDF.index[i+h],
                'price': p_fut,
                'lo': lo, 'hi': hi, 'hit': hit,
                'miss_type': miss_type,
                'residuals': residuals,
                'dw': dw
            })

        if resultsBT:
            res_df = pd.DataFrame(resultsBT).set_index('t')
            all_market_results[market_name] = res_df
            acc = res_df['hit'].mean() * 100
            
            summary_report.append({
                'Market': market_name[:40],
                'Accuracy': round(acc, 2),
                'Points': len(res_df),
                'Miss up': len(res_df[res_df['miss_type'] == 'up']),
                'Miss down': len(res_df[res_df['miss_type'] == 'down']),
            })
            print(f"      Accuracy: {acc:.2f}%")

    except Exception as e:
        print(f"Error on {market_name}: {e}")

# Results
report_df = pd.DataFrame(summary_report)
if not report_df.empty:
    print(report_df.to_string(index=False))
    avg_acc = report_df['Accuracy'].mean()
    print(f"\nAverage accuracy: {avg_acc:.2f}%")

# Plot
n_markets_per_row = 2 
cols_per_market = 2
total_cols = n_markets_per_row * cols_per_market # 4 columns
max_rows = 3
per_page = max_rows * n_markets_per_row # 6 markets / page

items = list(all_market_results.items())
n_markets = len(items)
pages = math.ceil(n_markets / per_page)

for page in range(pages):
    start = page * per_page
    end = start + per_page
    subset = items[start:end]
    
    current_rows = math.ceil(len(subset) / n_markets_per_row)
    
    fig, axes = plt.subplots(current_rows, total_cols, figsize=(20, 3 * current_rows), squeeze=False)
    
    for i, (m_name, bt_df) in enumerate(subset):
        row_idx = i // n_markets_per_row
        col_offset = (i % n_markets_per_row) * cols_per_market
        
        ax_left = axes[row_idx, col_offset]
        ax_right = axes[row_idx, col_offset + 1]

        # Title
        fig.text(0.25 + (0.5 * (i % n_markets_per_row)), 0.98 - (row_idx * 0.98 / current_rows), f"HAR-RV | {m_name[:30]}...", ha='center', fontsize=8)
        
        # Left: price and confidence interval
        acc = bt_df['hit'].mean() * 100
        ax_left.plot(bt_df.index, bt_df['price'], color='black', lw=0.5, label='Price', zorder=3)
        ax_left.fill_between(bt_df.index, bt_df['lo'], bt_df['hi'], color='orange', alpha=0.2, step='post')
        
        miss_up = bt_df[bt_df['miss_type'] == 'up']
        ax_left.scatter(miss_up.index, miss_up['price'], color='red', s=3, label='Miss up')
        miss_down = bt_df[bt_df['miss_type'] == 'down']
        ax_left.scatter(miss_down.index, miss_down['price'], color='limegreen', s=3, label='Miss down')
        
        ax_left.set_title(f"Acc: {acc:.1f}%", fontsize=8, pad=5)
        ax_left.legend(prop={'size': 6}, loc='upper left')

        # Right: residuals and Durbin-Watson
        ax_right.plot(bt_df.index, bt_df['residuals'], color='black', lw=0.5, zorder=3)
        ax_right.axhline(0, color='red', linestyle='-', lw=0.8)
        
        ax_dw = ax_right.twinx()
        ax_dw.plot(bt_df.index, bt_df['dw'], color='royalblue', lw=0.5, alpha=0.6, zorder = 3)
        ax_dw.fill_between(bt_df.index, 1.5, 2.5, color='royalblue', alpha=0.1)
        ax_dw.set_ylim(0, 4)
        ax_dw.tick_params(axis='y', labelsize=7, color='royalblue')
        
        ax_right.set_title(f"Residuals & Durbin-Watson", fontsize=8, pad=5)

        # Appearance
        for ax in [ax_left, ax_right]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(alpha=0.2)

    # Clearing empty axes
    for j in range(i + 1, current_rows * n_markets_per_row):
        r = j // n_markets_per_row
        c = (j % n_markets_per_row) * cols_per_market
        axes[r, c].axis('off')
        axes[r, c+1].axis('off')

    plt.tight_layout()#rect=[0, 0, 1, 0.95])
    #fig.subplots_adjust(wspace=0.4)
    plt.show()