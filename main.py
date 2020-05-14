import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def getdata(file):
    """
    load dataframe from file, preprocess it
    """
    df = pd.read_csv(file)
    df = df.rename(columns = {
        'dateRep' : 'date',
        'countriesAndTerritories' : 'country',
        'countryterritoryCode' : 'countryCode', 
        'geoID' : 'geoId',
        'popData2018' : 'pop'
        })
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    countries = ['Netherlands', 'Belgium', 'Italy', 'Sweden', 'Denmark', 'Norway', 'Spain', 'United_Kingdom', 'Germany', 'Romania']
    df = df[df.country.isin(countries)]
    df = df.sort_values('date')
    return df

def filtercountry(df, country):
    """
    returns the df filtered for a specific country
    """
    mask = df.country.eq(country)
    return df[mask]

def minmax(x):
    """
    min max scaling between 0 and 1
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def ratioincrease(x, timelag):
    """
    transforms each element into the ratio increase of it compared to its correspondent in timelag before
    """
    r = []
    for i in range(timelag, len(x)):
        ratio = (x[i] - x[i - timelag]) / x[i - timelag]
        r.append(ratio)

    r = np.array(r)
    return r

def process(df):
    """
    return time ticks (days) and cases for a df, after preprocessing them
    """
    dates = df['date']
    cases = df['cases'].values
    cases = cases[cases > 0 ]
    # cases = np.cumsum(cases)
    # use preprocess cases functions here
    # cases = ratioincrease(cases, timelag = 1)
    # cases = minmax(cases)

    return range(len(cases)), cases

def plotcountry(df, country, ax, color = None):
    """
    plot country spread
    """
    df = filtercountry(df, country)
    dates, cases = process(df)
    ax.plot(dates, cases, '-', color = color,)
    return ax

def explore(df):
    print(len(df))
    print(df)
    print(df.country.unique())
    print(filtercountry(df, 'Netherlands'))
    aggs = df.groupby('country').agg({
        'cases' : ['mean', 'median', 'min', 'max', 'sum']
        })
    aggs = aggs.sort_values(('cases', 'sum'))
    print(aggs)

if __name__ == '__main__':
    data = 'data/timeseries.csv'
    df = getdata(data)
    toexplore = False
    countries = ['Netherlands', 'Belgium', 'Italy', 'Sweden', 'Denmark', 'Norway', 'Spain', 'United_Kingdom', 'Germany', 'Romania']
    if toexplore:
        explore(df)
    toplot = True
    if toplot:
        fig, ax = plt.subplots()
        c1 = 'Netherlands'
        c2 = 'Belgium'
        plotcountry(df, c1, ax, color = 'blue')
        plotcountry(df, c2, ax, color = 'red')
        plotcountry(df, 'Italy', ax, color = 'green')
        # for c in countries:
        #     plotcountry(df, c, ax)
        plt.xticks(rotation = 90)
        plt.axhline(color = 'grey', linestyle = '--')
        plt.show()
