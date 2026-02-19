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
        df['returns2'] = df['logReturn'].dropna() ** 2 # Scaled returns for arch stability

        # Application of HAR-RV settings
        df['RVD'] = df['returns2'].rolling(hd, min_periods=1).mean()
        df['RVW'] = df['returns2'].rolling(hw, min_periods=1).mean()
        df['RVM'] = df['returns2'].rolling(hm, min_periods=1).mean()

        harDF = df[['RVD','RVM', 'RVW', 'returns2']].dropna(subset=['returns2'])

        # Backtesting
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
            pred_R2 = max(0, model.predict(current_features)[0])
            
            sigma = np.sqrt(pred_R2)
            p_now = df['p'].iloc[i]
            p_fut = df['p'].iloc[i+h]
            
            lo, hi = max(0, p_now * (1 - sigma * 1.96)), min(100, p_now * (1 + sigma * 1.96))
            hit = lo <= p_fut <= hi
            miss_type = None

            if not hit:
                miss_type = 'up' if p_fut > hi else 'down'
            
            resultsBT.append({
                't': harDF.index[i+h],
                'price': p_fut,
                'lo': lo,
                'hi': hi,
                'hit': hit,
                'miss_type': miss_type
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
                'Miss down': len(res_df[res_df['miss_type'] == 'down'])

            })
            print(f"   Accuracy: {acc:.2f}%")

    except Exception as e:
        print(f"Error on {market_name}: {e}")

# Results
report_df = pd.DataFrame(summary_report)
if not report_df.empty:
    print(report_df.to_string(index=False))
    avg_acc = report_df['Accuracy'].mean()
    print(f"\nAverage accuracy: {avg_acc:.2f}%")

# Plot
if all_market_results:
    items = list(all_market_results.items())
    max_rows, cols = 3, 4
    per_page = max_rows * cols
    pages = math.ceil(len(items) / per_page)

    for page in range(pages):
        subset = items[page*per_page : (page+1)*per_page]
        rows = math.ceil(len(subset) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
        axes = np.array(axes).flatten()

        for i, (m_name, bt_df) in enumerate(subset):
            ax = axes[i]
            acc = bt_df['hit'].mean() * 100

            ax.plot(bt_df.index, bt_df['price'], color='black', lw=1, label='Price', zorder=3)
            ax.fill_between(bt_df.index, bt_df['lo'], bt_df['hi'], color='orange', alpha=0.2, label='95% interval', step='post')
            
            miss_up = bt_df[bt_df['miss_type'] == 'up']
            ax.scatter(miss_up.index, miss_up['price'], color='red', s=10, zorder=5, label='Miss up')
            
            miss_down = bt_df[bt_df['miss_type'] == 'down']
            ax.scatter(miss_down.index, miss_down['price'], color='limegreen', s=10, zorder=5, label='Miss down')
            
            ax.set_title(f"{m_name[:30]}...\nAcc: {acc:.1f}%", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax.grid(alpha=0.2)
            ax.legend(prop={'size': 7})

        for j in range(len(subset), len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.show()
