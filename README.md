# VolatilityAnalysis
This project uses various volatility models in order to measure observed volatility and forecast price interval for Polymarket's Politics and Sports markets.

## Global analysis of markets

#### 1. ```utils.py```
In order to pursue volatility analysis, we focus on two types of Polymarket prediction markets: Sports and Politics markets. These are the most common and most popular markets on this platform.
Among these categories, we add selection criterias: 
- **High Volume**: we used high-volume markets (1M+ volume) for this analysis. This setting can be modified by user input for each program.
- **Sufficient price change**: in order to measure volatility, we need price movement. Hence, we filter out markets whose price haven't moved in the last month, week, or days.
With these filters, between 100 and 150 markets should remain eligible.

#### 2. ```returnsDistribution.py```
Before applying volatility models to our dataset, we need to understand its caracteristics.
![GARCHOutput](assets/StatisticalDistribution.png)
Log-returns on high-volume prediction markets are super-leptokurtik: they have a highly positive Kurtosis. Thus, using a normal distribution for our GARCH analysis wouldn't be accurate. From now on, we'll use the *Generalized Error Distribution* (GED) of the garch library to fit our models.

#### 3. ```skewSmile.py```
The statistical distribution of log-returns already gives an approximation of the skewness of the data. In order to observe its distribution and its eventual smile or skew, we can plot observed volatility according to price using a simple ARCH(1) model. We'll use more complex models later on.
![Skew](assets/VolatilitySkew.png)
This graph does not necessarily indicates volatility skewness, rather a logical mechanism. It reflects how low prices have higher relative changes, and high prices have lower relative changes, with a plateau from 30c to 70c.

#### 3. ```getData.py```
Downloads .csv price history of eligible markets in data/Politics and data/Sports.

## GARCH models

