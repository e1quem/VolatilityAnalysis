# VolatilityAnalysis
This project uses volatility models (GARCH, REGARCH, HAR-RV, Rogers-Satchell, Ornstein-Uhlenbeck process) to measure observed volatility and forecast price intervals for Polymarket's Politics and Sports markets.

## Global analysis of markets

#### 1. ```utils.py```
In order to pursue volatility analysis, we focus on two types of Polymarket prediction markets: Sports and Politics markets. These are the most common and most popular markets on this platform.
Among these categories, we add selection criterias: 
- **High Volume**: we used high-volume markets (1M+ volume) for this analysis. This setting can be modified by user input for each program.
- **Sufficient price change**: in order to measure volatility, we need price movement. Hence, we filter out markets whose price haven't moved in the last month, week, or day.

With these filters, between 100 and 150 markets should remain eligible.

#### 2. ```returnsDistribution.py```
Before applying volatility models to our dataset, we need to understand its caracteristics.
![GARCHOutput](assets/StatisticalDistribution.png)
Log-returns on high-volume prediction markets are super-leptokurtik: they have a highly positive Kurtosis. Thus, using a normal distribution for our GARCH analysis wouldn't be accurate. From now on, we'll use the *Generalized Error Distribution* (GED) of the garch library to fit our models, for adaptative sharp peak and larger tails.

#### 3. ```skewSmile.py```
The statistical distribution of log-returns already gives an approximation of the skewness of the data. In order to observe its distribution and its eventual smile or skew, we can plot observed volatility according to price using a simple ARCH(1) model. We'll use more complex models later on.
![Skew](assets/VolatilitySkew.png)
This graph does not necessarily indicates volatility skewness, but rather a logical mechanism. It reflects how low prices have higher relative changes, and high prices have lower relative changes, with a plateau from 30c to 70c.

#### 3. ```getData.py```
Downloads .csv price history of eligible markets in data/Politics and data/Sports folders.

## GARCH models

#### 1. ```GARCHbacktest.py````
This file uses GARCH volatility models to backtest volatility forecasts on individual markets (requires a slug). The model is fitted on training points, and the volatility forecast it produces is then used to define 95% confidence price interval forecasts for the next aggregated time period. Then, 10m price is used to count hits (price is in the interval) and misses (price is outside of the predicted price range).

We do not fit one model for all data points. For each step, we fit 6 models: ARCH(1), ARCH(2), GARCH(1,1), GARCH(2,1), TARCH(1,1,1), TARCHC(2,2,1), before comparing their BIC and using the model with the lowest one for the forecast.

![GARCH](assets/GARCH.png)

#### 2. ```GARCHbacktests.py```
This file has the same basic principe as ```GARCHbacktest.py```but iterates on multiple high-volume markets. 
To reduce computing power usage, models are reduced to ARCH(1), GARCH(1,1) and TARCH(1,1,1), since these were the three ones regularly obtaining the lowest BICs on these markets. 

![GARCHs](assets/GARCH.png)

#### 3. ```RogersSatchell.py````

GARCH models can use exogeneous variables. It would have been useful to use data such as volume in the model. However, such historical data is not obtainable via Polymarket API, obtaining it would require constant Websocket connection and data hoarding for a few weeks. This is a potential future projects to bypass the 10m maximumum fidelity, and the lack of other variables such as volume.

However, we can still use more information by transforming our aggregated data into candles, which carries more information than simple closing prices.

In this sense, we use **Rogers & Satchell (1991)** formula: 
$$\sigma^2_{rs}=\frac{1}{n}\sum\limits^2_{i=1}(\log(\frac{H_i}{C_i})\log(\frac{H_i}{O_i}+\log\frac{L_i}{C_i}\log\frac{L_i}{O_i}))$$
as an exogeneous variable vor GARCH(1,1) and TARCH(1,1,1). 

This added factor only adds complexity to our models and does not improve them on these prediction markets. They never obtain the lowest BIC compared to their simpler counterparts.


#### 4. Upcoming

- Ornstein-Uhlenbeck process
- REGARCH

## HAR-RV model

