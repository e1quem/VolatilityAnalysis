import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings
import math
import os

warnings.filterwarnings("ignore")

# Config
base_path = os.path.expanduser('~/Desktop/volAnalysis/data/')

# Uses getData.py downloaded price history for stable comparison
# Populating the /data folder is required before launching this file
# Chosing market type
market_type = input("\033[1mSelect type\033[0m: Politics (1) | Sports (2) | Both (3): ")
folders = []
if market_type == '1':
    folders.append(os.path.join(base_path, 'Politics'))
elif market_type == '2':
    folders.append(os.path.join(base_path, 'Sports'))
elif market_type == '3':
    folders.append(os.path.join(base_path, 'Politics'))
    folders.append(os.path.join(base_path, 'Sports'))
else:
    print("Invalid type")
    exit()

paths = []
for folder in folders:
    if os.path.exists(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        paths.extend(files)
print(f"Found {len(paths)} markets.")

num_markets = int(input(f"\033[1mAmount of markets to analyze\033[0m: "))
markets = paths[:num_markets]

# Global performance
global_hits = 0
global_total_points = 0
all_results = {}

# Backtesting
for file_path in markets:
    file_name = os.path.basename(file_path)
    print(f"\n\033[3m{file_name}\033[0m...")
    dfFull = pd.read_csv(file_path)

    # Dataframe creation
    dfFull['t'] = pd.to_datetime(dfFull['t'])
    dfFull['p'] = (dfFull['p']).round(1)
    dfFull.set_index('t', inplace=True)
    duration = (dfFull.index[-1] - dfFull.index[0]).total_seconds() / 86400

    # Defining price aggregation according to price history length
    if duration < 1:
        aggregate = '10min'
    elif 1 <= duration < 3:
        aggregate = '30min'
    elif 3 <= duration < 7:
        aggregate = '1H'
    else:
        aggregate = '4H'

    df = dfFull['p'].resample(aggregate).last().ffill().to_frame()
    df['logReturn'] = np.log(df['p'] / df['p'].shift(1))

    # Sliding window to calculate realized volatility
    windowSize = 20 
    df['realized_vol'] = df['logReturn'].rolling(window=windowSize).std()

    trainingPoints = 40
    if len(df) <= trainingPoints + 10:
        print("Not enough data for significant backtesting.")
    else:
        resultsBT = []
    

        for i in range(windowSize, len(df) - 1):

            train_vol = df['realized_vol'].iloc[:i].dropna()
            if len(train_vol) < 10:
                continue

            # Ajusting OU process on realized volatility
            X = train_vol.values

            if len(X) < 10 or np.all(X == X[0]): # Security check for constant data
                continue

            X_lag = X[:-1]
            X_curr = X[1:]

            # Linear regression to estimate OU parameters
            X_lag_const = sm.add_constant(X_lag)

            try:
                model = sm.OLS(X_curr, X_lag_const)
                res = model.fit()

                if len(res.params) < 2:
                    raise ValueError("Regression failed to find phi")

                phi = res.params[1]
                c = res.params[0]
                sigma_u2 = res.resid.var()

                if 0 < phi < 1:
                    kappa = -np.log(phi)  # Mean reversing rate
                    mu = c / (1 - phi)     # Long term level
                    sigma_ou = np.sqrt(sigma_u2) # OU process volatility
                else:
                    # Non-stationary regime
                    kappa = 1e-4  
                    mu = train_vol.mean()
                    sigma_ou = np.sqrt(sigma_u2)

            except Exception:
                kappa = 1e-4
                mu = X.mean()
                sigma_ou = X.std() if len(X) > 1 else 0.01

            # Volatility t+1 forecast
            last_vol = train_vol.iloc[-1]
            forecast_vol = mu + (last_vol - mu) * np.exp(-kappa)

            # Price interval calaculation
            current_price = df['p'].iloc[i]
            k = 1.96 
            lo = max(0, current_price * np.exp(-k * forecast_vol))
            hi = min(100, current_price * np.exp(k * forecast_vol))

            # Results
            resultsBT.append({
                't': df.index[i+1],
                'price': current_price,
                'lo': lo,
                'hi': hi,
                'kappa': kappa,
                'mu': mu,
                'sigma': sigma_ou
            })

        # Analysis
        if not resultsBT:
            continue
        
        # Merging
        dfBT = pd.DataFrame(resultsBT).set_index('t')
        dfFull = dfFull.sort_index()
        dfBT = dfBT.sort_index()
        final = pd.merge_asof(dfFull[['p']], dfBT, left_index=True, right_index=True, direction='backward')
        final.dropna(subset=['lo', 'hi'], inplace=True)
        final['hit'] = (final['p'] >= final['lo']) & (final['p'] <= final['hi'])

        accuracy = final['hit'].mean() * 100
        hits = final['hit'].sum()
        total_points = len(final)
        miss_up = (final['p'] > final['hi']).sum()
        miss_down = (final['p'] < final['lo']).sum()
        avg_kappa = final['kappa'].mean()
        avg_mu = final['mu'].mean()
        avg_sigma = final['sigma'].mean()

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Hits:           {hits} ({hits/total_points*100:.2f}%)")
        print(f"Miss up:        {miss_up} ({miss_up/total_points*100:.2f}%)")
        print(f"Miss down:      {miss_down} ({miss_down/total_points*100:.2f}%)")
        print(f"Mean Reversion: {avg_kappa:.4f}")
        print(f"Long-term Vol:  {avg_mu:.4f}")
        print(f"Vol of Vol:     {avg_sigma:.4f}")

        global_hits += hits
        global_total_points += total_points
        all_results[file_name] = final.copy()

# Global summary
print(f"\n\033[1mGlobal Summary ({len(markets)} markets)\033[0m")
if global_total_points > 0:
    print(f"Global Accuracy: {(global_hits / global_total_points) * 100:.2f}%")
else:
    print("Not enough data for global summary.")

# Graph
max_rows = 3
cols = 4
per_page = max_rows * cols
pages = math.ceil(len(all_results) / per_page)
items = list(all_results.items())

for page in range(pages):
    start = page * per_page
    end = start + per_page
    subset = items[start:end]

    rows = math.ceil(len(subset) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()

    for i, (m_name, final) in enumerate(subset):
        ax = axes[i]
        acc = (final['hit'].sum() / len(final)) * 100

        ax.plot(final.index, final['p'], color='black', linewidth=1, label='Price', zorder=3)
        ax.fill_between(final.index, final['lo'], final['hi'], color='orange', alpha=0.2, label='95% interval')

        # Misses
        miss_up = final[final['p'] > final['hi']]
        ax.scatter(miss_up.index, miss_up['p'], color='red', s=10, zorder=5, label='Miss up')

        miss_down = final[final['p'] < final['lo']]
        ax.scatter(miss_down.index, miss_down['p'], color='limegreen', s=10, zorder=5, label='Miss down')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.set_title(f"{m_name[:40]}...\nAcc: {acc:.1f}%", fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
