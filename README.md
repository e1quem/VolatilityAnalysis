# VolatilityAnalysis
This project uses volatility models (GARCH, REGARCH, HAR-RV, Rogers-Satchell, Ornstein-Uhlenbeck process) to measure observed volatility and forecast price intervals in a sandbox environment: Polymarket's Politics and Sports markets.

## How to run

## Global analysis of markets

#### 1. [utils.py](./utils.py)
In order to pursue volatility analysis, we focus on two types of Polymarket prediction markets: Sports and Politics markets. These are the most common and most popular markets on this platform.
Among these categories, we add selection criteria: 
- **High Volume**: we used high-volume markets (1M+ volume) for this analysis. This setting can be modified by user input for each program.
- **Sufficient price change**: in order to measure volatility, we need price movement. Hence, we filter out markets whose price haven't moved in the last month, week, or day.

With these filters, between 100 and 150 markets should remain eligible.

#### 2. [returnsDistribution.py](./returnsDistribution.py)
Before applying volatility models to our dataset, we need to understand its characteristics.
![GARCHOutput](assets/StatisticalDistribution.png)
Log-returns on high-volume prediction markets are highly leptokurtik: they have a highly positive Kurtosis. Thus, using a normal distribution for our GARCH analysis wouldn't be accurate. From now on, we'll use the *Generalized Error Distribution* (GED) of the garch library to fit our models, for adaptative sharp peak and larger tails.

#### 3. [skewSmile.py](./skewSmile.py)
The statistical distribution of log-returns already gives an approximation of the skewness of the data. In order to observe its distribution and its eventual smile or skew, we can plot observed volatility according to price using a simple ARCH(1) model. We'll use more complex models later on.
![Skew](assets/VolatilitySkew.png)
This graph does not necessarily indicate volatility skewness, but rather a logical mechanism. It reveals how low prices have higher relative changes, and high prices have lower relative changes, with a plateau from 30c to 70c.

#### 4. [getData.py](./getData.py)
Downloads .csv price history of eligible markets in data/Politics and data/Sports folders.

## GARCH models

#### 1. [GARCHbacktest.py](GARCH/GARCHbacktest.py)
This file uses GARCH volatility models to backtest volatility forecasts on individual markets (requires a slug). The model is fitted on training points, and the volatility forecast it produces is then used to define 95% confidence price interval forecasts for the next aggregated time period. Then, 10m price is used to count hits (price is in the interval) and misses (price is outside of the predicted price range).

We do not fit one model for all data points. For each step, we fit 6 models: ARCH(1), ARCH(2), GARCH(1,1), GARCH(2,1), TARCH(1,1,1), TARCH(2,2,1), before comparing their BIC and using the model with the lowest one for the forecast.

![GARCH](assets/GARCH.png)

#### 2. [GARCHbacktests.py](GARCH/GARCHbacktests.py)
This file has the same basic principle as ```GARCHbacktest.py``` but iterates on multiple high-volume markets. 
To reduce computing power usage, models are reduced to ARCH(1), GARCH(1,1) and TARCH(1,1,1), since these were the ones regularly obtaining the lowest BICs on these markets. 

![GARCHs](assets/GARCHs.png)

#### 3. [RogersSatchell.py](GARCH/RogersSatchell.py)

GARCH models can use exogenous variables. It would have been useful to use data such as volume in the model. However, such historical data is not obtainable via Polymarket API, obtaining it would require constant Websocket connection and data hoarding for a few weeks. This is a potential future projects to bypass the 10m maximumum fidelity, and the lack of other variables such as volume.

However, we can still use more information by transforming our aggregated data into candles, which carries more information than simple closing prices.

In this sense, we use **Rogers & Satchell (1991)** formula

$$\sigma^2_{rs}=\frac{1}{n}\sum^n_{i=1} \left( \log\left(\frac{H_i}{C_i}\right) \log\left(\frac{H_i}{O_i}\right) + \log\left(\frac{L_i}{C_i}\right) \log\left(\frac{L_i}{O_i}\right) \right)$$

as an exogenous variable for GARCH(1,1) and TARCH(1,1,1). 

This added factor only adds complexity to our models and does not improve them on these prediction markets. They never obtain the lowest BIC compared to their simpler counterparts.


#### 4. Upcoming

- Ornstein-Uhlenbeck process
- REGARCH

## HAR-RV model

Since our data has a relatively high granularity (10m), using GARCH models is sub-optimal: these models are usually made to perform on daily prices for extended periods of time.

For intraday predictions, we use the HAR-RV model. Its standard version takes into account different time horizons: monthly, weekly, and daily volatility. Its simple autoregressive structure is simple but takes into account different time periods and has a large memory.

#### 1. [ratioTesting.py](HAR-RV/ratioTesting.py)

This file allows to compare different ratios on individual markets (requires market slug). The standard ratio is 1:7:30. User can add ratios it wants to test in the file. 1 unit corresponds to a 10m time period.

The rankings it produces was used to define ratios to use on HAR-RV backtests.

#### 2. [HARbacktest.py](HAR-RV/HARbacktest.py)

This file follows the same structure as ```GARCHbacktest.py```but uses HAR-RV instead of GARCH models in order to forecast volatility. HAR-RV ranges are adaptative according to the length of available data.

By default, we chose: 
- a reactive 6:18:36 (1h:3h:6h) for markets with less than a day of price history
- a longer 24:72:144 (4h:12h:24h) for all other markets.

![HAR-RV](assets/HAR-RV.png)

With HAR-RV, the 95% confidence interval is tighter. Since it operates on more granular data (10m compared to sometimes 4h for the GARCH method), this tight can maintain an accuracy close the 95% with more misses but also more hits.

#### 3. [HARbacktests.py](HAR-RV/HARbacktests.py)

Same structure as ```HARbacktest.py``` but allows to test multiple markets at once with type and volume filtering.


## Findings