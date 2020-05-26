import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

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

def getmeasures(file):
    """
    load measures data from csv file
    """
    df = pd.read_csv(file)
    df = df.rename(columns = {
        'Country' : 'date'
        })
    df = df.iloc[1:]
    print(df.date.unique())
    # df['date'] = pd.to_datetime(df['date'])
    df['date'] = pd.to_datetime(df['date'], format='%d/%m')
    return df

def filtercountry(df, country):
    """
    returns the df filtered for a specific country
    """
    mask = df.country.eq(country)
    return df[mask]

def filtermeasures(df, country):
    """
    returns the dfmeasures filtered for a specific country
    """
    mask = ~df[country].isna()
    return df[['date', country]][mask]

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
    cases = np.cumsum(cases)
    # use preprocess cases functions here
    # cases = ratioincrease(cases, timelag = 1)
    cases = minmax(cases)

    return range(len(cases)), cases


def plotcountry(df, country, ax, color = None):
    """
    plot country spread
    """
    df = filtercountry(df, country)
    dates, cases = process(df)
    ax.plot(dates, cases, '-', color = color, label = country)
    return ax

def explore(df):
    print(df)
    print('Total lines ', len(df))
    print(df.country.unique())
    print('Total countries ', len(df.country.unique()))
    aggs = df.groupby('country').agg({
        'cases' : ['mean', 'median', 'min', 'max', 'sum']
        })
    aggs = aggs.sort_values(('cases', 'sum'))
    print('Aggregation ####')
    print(aggs)

def newimageidx(output_dir):
    idxfile = os.path.join(output_dir, '.idx')
    if os.path.exists(idxfile):
        with open(idxfile, 'r') as f:
            idx = int(f.read())
        idx += 1
    else:
        idx = 0
    with open(idxfile, 'w') as f:
        f.write(str(idx))
    return idx


if __name__ == '__main__':
    toexplore = True
    toplot = False

    data = 'data/timeseries.csv'
    measuresdata = 'data/measures_start.csv'
    output_dir = 'output'
    countries = ['Netherlands', 'Belgium', 'Italy', 'Sweden', 'Denmark', 'Norway', 'Spain', 'United_Kingdom', 'Germany', 'Romania']
    ldc = [ 'Italy', 'Spain', 'United_Kingdom', 'Romania']

    sdc = [c for c in countries if c not in ldc]
    df = getdata(data)
    mdf = getmeasures(measuresdata)
    if toexplore:
        explore(df)
        print('Measures')
        # mdf = filtermeasures(mdf, 'Netherlands')
        print(mdf)
    if toplot:
        fig, ax = plt.subplots()
        for c in countries:
            color = 'red' if c in ldc else 'blue'
            plotcountry(df, c, ax, color = color)
        plt.legend()
        plt.xlabel('Days since first positive test')
        plt.ylabel('Positive tests')
        plt.title('Positive tests ')
        idx = newimageidx(output_dir)
        plt.savefig(f'{output_dir}/out{idx}.png')
        plt.show()
