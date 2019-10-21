from pandas import read_csv
from pandas import datetime
 
def parser(x):
	return datetime.strptime(x, '%Y-%m')


csv_name = 'Rogongon_trimmed.csv'
 
series = read_csv(csv_name)
series['date_and_time'] = series['YEAR'].map(str) + '-'+ series['MONTH'].map(str) + '-'+ series['DAY'].map(str) + ' ' + series['TIME'].map(str)


date_and_time = series['date_and_time']
RAINFALL = series['RAINFALL']

new_dataset = series[['date_and_time','RAINFALL']]

new_name = 'processed_rogongon.csv'
new_dataset.to_csv(new_name)


series = read_csv(new_name, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

upsampled = series.resample('D')
print(upsampled.head(32))