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
    print(x)
    print(len(x))
    for i in range(timelag, len(x)):
        ratio = x[i] / x[i - timelag]
        ratio = ratio - 1
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
    # use preprocess cases functions here
    cases = ratioincrease(cases, timelag = 1)

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
    print(df[df.cases < 0])
    aggs = df.groupby('country').agg({
        'cases' : ['mean', 'median', 'min', 'max']
        })
    print(aggs)


if __name__ == '__main__':
    data = 'data/timeseries.csv'
    df = getdata(data)
    toexplore = True
    if toexplore:
        explore(df)
    toplot = False
    if toplot:
        fig, ax = plt.subplots()
        plotcountry(df, 'Romania', ax)
        plotcountry(df, 'Netherlands', ax)
        # plotcountry(df, 'Italy', ax, color = 'red')
        plt.xticks(rotation = 90)
        plt.axhline(color = 'grey', linestyle = '--')
        plt.show()
