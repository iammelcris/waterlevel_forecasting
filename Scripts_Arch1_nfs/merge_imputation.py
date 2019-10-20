import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



series = pd.read_csv('merged.csv')


date_and_time = series['DATETTIME'].tolist()

rainfall = series['RAINFALL_ROGONGON']*1000
waterlevel = rainfall
waterlevel = waterlevel.tolist()

values = waterlevel
timestamps = pd.to_datetime(date_and_time)

ts = pd.Series(values, index=timestamps)
ts = ts.resample('30T').mean()

ts.interpolate(method='spline', order=3).plot()
ts.interpolate(method='time').plot()
ts.interpolate(method='linear', inplace=True)


print(str(ts))
ts.columns = ['DATETIME', 'WATERLEVEL', 'RAINFALL']
ts.to_csv('merged_imputation3.csv')



