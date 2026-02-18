from datetime import datetime, timezone
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
import pandas as pd
import numpy as np
import warnings 
import requests
import socket
import json
import os

warnings.filterwarnings("ignore")

# Used downloaded price history from getData.py
base_path = os.path.expanduser('~/Desktop/volAnalysis/data/')
type_input = input(f"\033[1mType of market to analyze\033[0m (1: Politics, 2: Sports): ")

if type_input == '1': 
    folder_path = os.path.join(base_path, 'Politics')
elif type_input == '2': 
    folder_path = os.path.join(base_path, 'Sports')
else:
    print("Invalid type")
    exit()

all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"Found {len(all_files)} markets in {folder_path}")

num_markets = int(input(f"\033[1mAmount of markets to analyze\033[0m: "))
files_to_analyze = all_files[:num_markets]

# Global performance
global_model_counts = {}
global_hits = 0
global_total_points = 0
market_success_rates = []

# Cette analyse devrait se faire dans la boucle, notamment pour la half life et la durée de la window
for file_name in files_to_analyze:
    print(f"\n\033[3m{file_name}\033[0m...")

    file_path = os.path.join(folder_path, file_name)
    dfFull = pd.read_csv(file_path)

    # Dataframe creation
    dfFull['t'] = pd.to_datetime(dfFull['t'])
    dfFull['p'] = (dfFull['p']).round(1)
    dfFull.set_index('t', inplace=True)

    duration = (dfFull.index[-1] - dfFull.index[0]).total_seconds() / 86400

    # Defining aggregation
    if duration < 1:
        aggregate = '10min'
    elif 1 <= duration < 3:
        aggregate = '30min'
    elif 3 <= duration < 7:
        aggregate = '1H'
    else:
        aggregate = '4H'

    df = dfFull['p'].resample(aggregate).last().ffill().to_frame()

    # Calculating returns and auto-scaling them
    df['logReturn'] = np.log(df['p'] / df['p'].shift(1))
    std_dev = df['logReturn'].std()
    if std_dev > 0:
        scale = 1 / std_dev
    else: 
        scale = 1
    ScaledReturns = df['logReturn'].dropna() * scale

    # Ornstein-Ulhenbeck analysis for volatility regime 
    # Using a 48 period (8h) volatility window.
    v = df['logReturn'].rolling(window=48).std()
    mu, theta, p_value_theta, sigma_v = 0, 0, 1.0, 0 # Initialize variables
    
    if len(v) > 5:
        # Properly align data
        temp_df = pd.DataFrame({
            'y': v.diff(),
            'x': v.shift(1)
        }).dropna()

        if len(temp_df) > 0:
            # Use aligned data
            y = temp_df['y']
            x = temp_df['x']
            x_with_const = sm.add_constant(x)
            
            # Fit model
            model = sm.OLS(y, x_with_const)
            results = model.fit()
            
            # Extract parameters
            if 'const' in results.params:
                a = results.params['const']
                b = results.params['x']

                theta = -b
                mu = a / theta if theta != 0 else v.mean()
                half_life = np.log(2) / theta if theta > 0 else np.inf
                p_value_theta = results.pvalues['x']
                sigma_v = v.std()
            else:
                print("Model failed to produce a constant. Skipping OU.")

    # Defining models specs
    specs = {
        'ARCH(1)': {'p': 1, 'o': 0, 'q': 0},
        'GARCH(1,1)': {'p': 1, 'o': 0, 'q': 1},
        'TARCH(1,1,1)': {'p': 1, 'o': 1, 'q': 1}}
    bicResults = {}


    # Backtesting
    trainingPoints = 40 
    if len(ScaledReturns) <= trainingPoints + 10:
        print("Not enough data for significant backtesting.")
    else:
        resultsBT = []

        for i in range(trainingPoints, len(ScaledReturns) - 1):
            train_data = ScaledReturns.iloc[:i]

            # For each step, selection of the best model using BIC comparison
            best_bic = float('inf')
            current_best_name = None
            
            # Fitting the models
            for name, prm in specs.items():
                try:
                    m_tmp = arch_model(train_data, p=prm['p'], o=prm['o'], q=prm['q'], dist='ged')
                    res_tmp = m_tmp.fit(update_freq=0, disp='off', show_warning=False)

                    # Comparing the BIC only if the model converges
                    if res_tmp.convergence_flag == 0 and res_tmp.bic < best_bic:
                        best_bic = res_tmp.bic
                        current_best_name = name
                except: continue

            # The best model of each step is used for the t+1 forecast
            if current_best_name:
                b_prm = specs[current_best_name]
                
                try:
                    model_final = arch_model(train_data, p=b_prm['p'], o=b_prm['o'], q=b_prm['q'], dist='ged')
                    fit_bt = model_final.fit(update_freq=0, disp='off', show_warning=False)
                    
                    f_bt = fit_bt.forecast(horizon=1)
                    
                    sigma_raw = np.sqrt(f_bt.variance.iloc[-1, 0]) / scale
                    current_price = df['p'].iloc[i]
                    next_price = df['p'].iloc[i+1]
                    
                    # Adaptive Ornstein-Ulhenbeck interval based on market regime and volatility mean-reversion
                    current_vol_rs = df['logReturn'].rolling(window=48).std().iloc[i]

                    z_vol = (current_vol_rs - mu) / sigma_v if sigma_v > 0 else 0
                    
                    if p_value_theta <= 0.05:
                        if z_vol < 0: k = 1.96 * (1 + abs(z_vol) * sigma_v)
                        else: k = 1.96 * (1 - (z_vol * theta))
                        k = np.clip(k, 1.645, 2.576)
                        # The interval stays between 90% and 99% confidence
                    else:
                        # Fallback to a 97.5% confidence interval is the p-value isn't significant
                        k = 2.24
                    
                    lo = max(0, current_price * np.exp(-k * sigma_raw))
                    hi = min(100, current_price * np.exp(k * sigma_raw))
                    
                    resultsBT.append({
                        't': df.index[i+1],
                        'lo': lo, 'hi': hi,
                        'k_val': k,
                        'model': current_best_name
                    })

                except: continue

        # Results
        if len(resultsBT) == 0: continue

        bt_df = pd.DataFrame(resultsBT).set_index('t')

        dfFull = dfFull.sort_index()
        bt_df = bt_df.sort_index()
        limits = bt_df[['lo', 'hi']]
        final_analysis = pd.merge_asof(dfFull[['p']], limits, left_index=True, right_index=True)
        final_analysis.dropna(inplace=True)

        final_analysis['hit_10m'] = (final_analysis['p'] >= final_analysis['lo']) & (final_analysis['p'] <= final_analysis['hi'])

        accuracy_10m = final_analysis['hit_10m'].mean() * 100
        print(f"Accuracy: {accuracy_10m:.2f}%")

        hits = final_analysis['hit_10m'].sum()
        miss_up = (final_analysis['p'] > final_analysis['hi']).sum()
        miss_down = (final_analysis['p'] < final_analysis['lo']).sum()

        total_points = len(final_analysis)
        
        # Half life
        freq_minutes = {'10min': 10, '30min': 30, '1H': 60, '4H': 240}.get(aggregate, 10)
        half_life = (half_life * freq_minutes) / 60

        if total_points > 0:
                
                # Model used during backtesting
                model_counts = bt_df['model'].value_counts()

                print(f"\n\033[1m10min results\033[0m")
                print(f"Points   : {total_points}")
                print(f"Hits     : {hits} ({hits/total_points*100:.2f}%)")
                print(f"Miss up  : {miss_up} ({miss_up/total_points*100:.2f}%)")
                print(f"Miss down: {miss_down} ({miss_down/total_points*100:.2f}%)")
                print(f"Half life: {half_life:.2}h")
                
                global_hits += hits
                global_total_points += total_points
                market_success_rates.append(accuracy_10m)
                
                for m_name in bt_df['model']:
                    global_model_counts[m_name] = global_model_counts.get(m_name, 0) + 1

print(f"\n\033[1mGlobal Summary ({len(files_to_analyze)} markets)\033[0m")

if global_total_points > 0:
    print(f"Accuracy: {(global_hits / global_total_points) * 100:.2f}%")
    
    print(f"\n\033[1mVolatility models (per point):\033[0m")
    sorted_models = sorted(global_model_counts.items(), key=lambda x: x[1], reverse=True)
    for m_name, count in sorted_models:
        print(f"{m_name.ljust(15)}: {count}")
else:
    print("Not enough data for global summary.")