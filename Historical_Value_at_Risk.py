import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr

#Import data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

#Portfolio performance
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(Time)
    return returns, std

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock+'.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)

returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns = returns.dropna() 

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

returns['portfolio'] = returns.dot(weights)

def historicalVar(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    
    #Function to convert the data if is not a pandas dataframe
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVar, alpha=alpha)

    else:
        raise TypeError("Expected returns to be a dataframe or series")

def historicalCVar(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the CVar for dataframe / series
    """
    if isinstance(returns, pd.Series):
        belowVar = returns <= historicalVar(returns, alpha=alpha)
        return returns[belowVar].mean()

#Function to convert the data if is not a pandas dataframe
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVar, alpha=5)

    else:
        raise TypeError("Expected returns to be a dataframe or series")

# 1 day 
Time = 1

VaR = -historicalVar(returns['portfolio'], alpha=5)*np.sqrt(Time)
CVaR = -historicalCVar(returns['portfolio'], alpha=5)*np.sqrt(Time)
pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

InitialInvestment = 100000
print("Expected Portfolio Return:      ", round(InitialInvestment*pRet, 2))
print("Value at Risk 95th CI     :      ", round(InitialInvestment*VaR, 2))
print("Conditional VaR 95th CI     :      ", round(InitialInvestment*CVaR, 2))


