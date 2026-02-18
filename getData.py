# Downloads 10 minute price history for Sports and Politics markets on Polymarket
# Filters markets depending on tag and historical price change

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import warnings 
import requests
import socket
import utils
import json

warnings.filterwarnings("ignore")

utils.ipv4()
markets, volume = utils.getMarkets()

Politics = [m for m in markets if any(tag in ["Politics", "Trump"] for tag in m['tags'])]
Sports = [m for m in markets if "Sports" in m['tags']]

print(f"Found \033[1m{len(Politics)}\033[0m Politics markets and \033[1m{len(Sports)}\033[0m Sports markets")

# Saving Politics and Sports markets as csv files in respective folders in the data folder
# Defining and creating output folders
(Path.home() / "Desktop/volAnalysis/data/Politics").mkdir(parents=True, exist_ok=True)
(Path.home() / "Desktop/volAnalysis/data/Sports").mkdir(parents=True, exist_ok=True)

for market in Politics:
    token = market['token']
    name = market['name'].replace('/', '_').replace(' ', '_')
    spread = market['spread']
    try:
        res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')
        dfHist = pd.DataFrame(res.json()['history'])
        dfHist['t'] = pd.to_datetime(dfHist['t'], unit='s')
        dfHist['p'] = (dfHist['p'] * 100).round(1)
        dfHist.set_index('t', inplace=True)
        outputFile = Path.home() / f"Desktop/volAnalysis/data/Politics/{name}_{spread}.csv"
        dfHist.to_csv(outputFile)
        print(f"Saved Politics market: {outputFile}")
    except Exception as e:
        print(f"Error for {name}: {e}")

for market in Sports:
    token = market['token']
    name = market['name'].replace('/', '_').replace(' ', '_')
    spread = market['spread']
    try:
        res = requests.get(f'https://clob.polymarket.com/prices-history?market={token}&interval=max&fidelity=10')
        dfHist = pd.DataFrame(res.json()['history'])
        dfHist['t'] = pd.to_datetime(dfHist['t'], unit='s')
        dfHist['p'] = (dfHist['p'] * 100).round(1)
        dfHist.set_index('t', inplace=True)
        outputFile = outputFile = Path.home() / f"Desktop/volAnalysis/data/Sports/{name}_{spread}.csv"
        dfHist.to_csv(outputFile)
        print(f"Saved Sports market: {outputFile}")
    except Exception as e:
        print(f"Error for {name}: {e}")