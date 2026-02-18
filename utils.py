import socket
import requests
import sys
import json

# Solving connection issues (Great Firewall)
def ipv4():
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        responses = old_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo
    pass

# Filtering logic to avoid "bond" markets
def bond(m):
    month = m.get('oneMonthPriceChange')
    week = m.get('oneWeekPriceChange')
    day = m.get('oneDayPriceChange')
    if month is not None:
        return abs(month) >= 0.01
    if week is not None:
        return abs(week) >= 0.002
    if day is not None:
        return abs(day) >= 0.001
    return False

def getMarkets():
    volume = input("\033[1mMinimum volume\033[0m (in millions): ")
    volume2 = float(volume)*1000000
    maxMarkets = int(input("\033[1mMarket limit\033[0m (max 500): "))

    # Obtaining market token
    tokenRequest = requests.get(f'https://gamma-api.polymarket.com/markets?limit={maxMarkets}&volume_num_min={volume2}&ascending=false&closed=false&include_tag=true')
    data = tokenRequest.json()

    markets = [
        {
            "name": m['question'],
            "token": json.loads(m['clobTokenIds'])[0],
            "spread": m['spread'] * 100,
            "tags": [tag['label'] for tag in m.get('tags', [])]
        }
        for m in data 
        if m.get('clobTokenIds') and bond(m)
    ]
    return markets, volume

def marketType(markets):
    Politics = [m for m in markets if any(tag in ["Politics", "Trump"] for tag in m['tags'])]
    Sports = [m for m in markets if "Sports" in m['tags']]

    if len(Politics) > 0 or len(Sports) > 0:
        print(f"Found \033[1m{len(Politics)}\033[0m Politics markets and \033[1m{len(Sports)}\033[0m Sports markets")
        type = input("\033[1mSelect type\033[0m: Politics (1) | Sports (2) | Both (3): ")
        type = float(type)
    else: 
        print(f"Couldn't find any volatile markets.")
        sys.exit()

    if type == 1:
        markets = Politics
        typeName = "Politics"
    elif type == 2:
        markets = Sports
        typeName = "Sports"
    else:
        markets = Politics + Sports
        typeName = "Sports & Politics"

    if len(markets) == 0:
        print(f"User selected an empty category")
        sys.exit()
    return markets, typeName