# ARIMAX

ARIMAX Code:

pip install pmdarima
Requirement already satisfied: pmdarima in c:\users\padra\anaconda3\lib\site-packages (2.0.2)
Requirement already satisfied: statsmodels>=0.13.2 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (0.13.2)
Requirement already satisfied: joblib>=0.11 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (1.1.0)
Requirement already satisfied: numpy>=1.21.2 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (1.21.5)
Requirement already satisfied: urllib3 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (1.26.9)
Requirement already satisfied: scipy>=1.3.2 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (1.7.3)
Requirement already satisfied: pandas>=0.19 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (1.4.2)
Requirement already satisfied: scikit-learn>=0.22 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (1.0.2)
Requirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (0.29.28)
Requirement already satisfied: setuptools!=50.0.0,>=38.6.0 in c:\users\padra\anaconda3\lib\site-packages (from pmdarima) (61.2.0)
Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\padra\anaconda3\lib\site-packages (from pandas>=0.19->pmdarima) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in c:\users\padra\anaconda3\lib\site-packages (from pandas>=0.19->pmdarima) (2021.3)
Requirement already satisfied: six>=1.5 in c:\users\padra\anaconda3\lib\site-packages (from python-dateutil>=2.8.1->pandas>=0.19->pmdarima) (1.16.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\padra\anaconda3\lib\site-packages (from scikit-learn>=0.22->pmdarima) (2.2.0)
Requirement already satisfied: patsy>=0.5.2 in c:\users\padra\anaconda3\lib\site-packages (from statsmodels>=0.13.2->pmdarima) (0.5.2)
Requirement already satisfied: packaging>=21.3 in c:\users\padra\anaconda3\lib\site-packages (from statsmodels>=0.13.2->pmdarima) (21.3)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\padra\anaconda3\lib\site-packages (from packaging>=21.3->statsmodels>=0.13.2->pmdarima) (3.0.4)
Note: you may need to restart the kernel to use updated packages.
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
data = pd.read_csv("Belfast-2015-2022-data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.resample('M')['O3', 'NO2',  'temp'].mean()
data.head()
O3	NO2	temp
Date			
2015-01-31	43.392857	32.464286	5.671429
2015-02-28	45.333333	28.714286	7.442857
2015-03-31	43.640000	32.520000	7.492000
2015-04-30	46.115385	31.730769	8.769231
2015-05-31	53.960000	20.240000	10.592000
# Transformations may be needed
result = adfuller(data['O3'])
​
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
​
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
ADF Statistic: -5.338256
p-value: 0.000005
Critical Values:
	1%: -3.502
	5%: -2.893
	10%: -2.583
train_data = data[:len(data)-12]
test_data = data[(len(data)-12):]
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
​
acf_original = plot_acf(train_data['O3'])
​
pacf_original = plot_pacf(train_data['O3'])


train_data
O3	NO2	temp
Date			
2015-01-31	43.392857	32.464286	5.671429
2015-02-28	45.333333	28.714286	7.442857
2015-03-31	43.640000	32.520000	7.492000
2015-04-30	46.115385	31.730769	8.769231
2015-05-31	53.960000	20.240000	10.592000
...	...	...	...
2021-08-31	41.666667	22.233333	13.336667
2021-09-30	42.285714	20.523810	12.161905
2021-10-31	50.117647	19.294118	10.152941
2021-11-30	47.068966	19.551724	9.165517
2021-12-31	38.433333	26.166667	8.616667
84 rows × 3 columns

test_data
O3	NO2	temp
Date			
2022-01-31	40.583333	27.083333	8.079167
2022-02-28	55.541667	18.333333	8.087500
2022-03-31	46.964286	30.000000	8.975000
2022-04-30	53.100000	14.400000	9.650000
2022-05-31	53.850000	13.050000	11.100000
2022-06-30	45.692308	16.461538	13.346154
2022-07-31	45.136364	16.227273	13.950000
2022-08-31	44.000000	18.600000	14.264000
2022-09-30	47.346154	15.769231	12.061538
2022-10-31	42.285714	21.357143	10.271429
2022-11-30	40.291667	26.166667	9.166667
2022-12-31	40.185185	28.333333	7.066667
exog_train_data = train_data[['NO2', 'temp']]
exog_test_data = test_data[['NO2', 'temp']]
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(train_data['O3'], trace=True,
                          supress_warnings=True)
stepwise_fit.summary()
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(0,0,0)[0] intercept   : AIC=516.558, Time=0.24 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=554.240, Time=0.03 sec
 ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=517.224, Time=0.06 sec
 ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=520.961, Time=0.05 sec
 ARIMA(0,0,0)(0,0,0)[0]             : AIC=880.001, Time=0.02 sec
 ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=515.609, Time=0.23 sec
 ARIMA(0,0,2)(0,0,0)[0] intercept   : AIC=514.035, Time=0.09 sec
 ARIMA(0,0,3)(0,0,0)[0] intercept   : AIC=515.504, Time=0.09 sec
 ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=515.014, Time=0.16 sec
 ARIMA(1,0,3)(0,0,0)[0] intercept   : AIC=517.500, Time=0.27 sec
 ARIMA(0,0,2)(0,0,0)[0]             : AIC=708.632, Time=0.09 sec

Best model:  ARIMA(0,0,2)(0,0,0)[0] intercept
Total fit time: 1.370 seconds
SARIMAX Results
Dep. Variable:	y	No. Observations:	84
Model:	SARIMAX(0, 0, 2)	Log Likelihood	-253.018
Date:	Wed, 15 Mar 2023	AIC	514.035
Time:	21:40:25	BIC	523.759
Sample:	01-31-2015	HQIC	517.944
- 12-31-2021		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
intercept	44.4520	1.219	36.472	0.000	42.063	46.841
ma.L1	0.7390	0.101	7.349	0.000	0.542	0.936
ma.L2	0.3116	0.098	3.185	0.001	0.120	0.503
sigma2	24.0319	4.594	5.232	0.000	15.029	33.035
Ljung-Box (L1) (Q):	0.07	Jarque-Bera (JB):	4.29
Prob(Q):	0.79	Prob(JB):	0.12
Heteroskedasticity (H):	1.75	Skew:	0.51
Prob(H) (two-sided):	0.15	Kurtosis:	2.58


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
model = ARIMA(endog=train_data['O3'], exog=exog_train_data, order=(0, 0, 2))
model_fit = model.fit()
pred = model_fit.predict(start=test_data.index[0], end=test_data.index[-1], exog=exog_test_data)
pred
2022-01-31    40.547271
2022-02-28    48.809711
2022-03-31    42.390101
2022-04-30    51.137038
2022-05-31    50.718285
2022-06-30    46.783026
2022-07-31    46.412776
2022-08-31    44.730360
2022-09-30    48.281836
2022-10-31    46.456810
2022-11-30    44.517657
2022-12-31    44.997810
Freq: M, Name: predicted_mean, dtype: float64
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_data['O3'], pred)
rmse = np.sqrt(mse)
​
print(f'mse - : {mse}')
print(f'rmse - : {rmse}')
mse - : 11.879230343029109
rmse - : 3.446625936046601
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['O3'], label='Actual O3')
ax.plot(pred, label='Predicted O3')
ax.set_xlabel('Year',fontsize=12)
ax.set_ylabel('O3 (V µg/m³)',fontsize=12)
ax.set_title('O3 ARIMAX Model for Belfast City Centre 2022', fontsize=20)
​
ax.legend()
plt.show()
