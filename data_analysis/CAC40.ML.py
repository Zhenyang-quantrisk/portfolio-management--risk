import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
name_mapping = {
    'MC.PA': 'lv',
    'OR.PA': 'oreal',
    'BNP.PA': 'bnp',
    'GLE.PA': 'sg',
    'CS.PA': 'axa',
    'ACA.PA': 'credit',
    'TTE.PA': 'total-e',
    'SAN.PA': 'sanofi',
    'SU.PA': 'schneider',
    'AIR.PA': 'airbus'
}
french_company = list(name_mapping.keys())

data = yf.download(french_company, start='2015-01-01',auto_adjust=True)
df=data['Close']
volume=data['Volume']

r5=df.pct_change(5)
r20=df.pct_change(20)
r60=df.pct_change(60)

ret=np.log(df).diff()
vol20=ret.rolling(20).std()
vol60=ret.rolling(60).std()

volume20=volume.rolling(20).mean()
volume_change20=volume/volume20 - 1

forward_return_5d=df.shift(-5)/df - 1
market_future_r5=forward_return_5d.mean(axis=1)
relative_r5=forward_return_5d.sub(market_future_r5,axis=0)
print(relative_r5)

dataset = pd.DataFrame({
    'r5': r5.stack(),
    'r20': r20.stack(),
    'r60': r60.stack(),
    'vol20': vol20.stack(),
    'vol60': vol60.stack(),
    'volume_change20': volume_change20.stack(),
    'target': relative_r5.stack()
})

dataset=dataset.dropna()
print(dataset.head(20))
dataset=dataset.reset_index()
print(dataset.head(20))

split_date='2023-01-01'
features=['r5','r20','r60','vol20','vol60','volume_change20']

train_data=dataset[dataset['Date']<split_date]
test_data=dataset[dataset['Date']>=split_date]
X_train=train_data[features]
Y_train=train_data['target']
X_test=test_data[features]

from sklearn.linear_model import LinearRegression,Ridge
model1=LinearRegression()
model1.fit(X_train,Y_train)
y_pred=model1.predict(X_test)
coef=pd.Series(model1.coef_,index=features)
print(f'Coef:{coef},intercept:{model1.intercept_}')

result1=test_data[['Date','Ticker','target']].copy()
result1['prediction']=y_pred
print(result1.head())

daily_ic=result1.groupby('Date').apply(lambda x:
                                       x['prediction'].corr(x['target'],
                                                            method='spearman'))
print(daily_ic.describe())

# Check efficency of factors
for factors in features:
    factors_ic=test_data.groupby('Date').apply(
        lambda x: x[factors].corr(x['target'],method='spearman')
    )
    print(f'{factors}:',factors_ic.describe())

corr_m=train_data[features].corr()
print(corr_m)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_x=scaler.fit_transform(X_train)
test_x=scaler.transform(X_test)
model2=Ridge()
model2.fit(train_x,Y_train)
y2_pred=model2.predict(test_x)

result2=test_data[['Date','Ticker','target']].copy()
result2['prediction']=y2_pred
daily_ic2=result2.groupby('Date').apply(
    lambda x: x['prediction'].corr(x['target'],method='spearman')
)
print(daily_ic2.describe())

from sklearn.ensemble import RandomForestRegressor
model3=RandomForestRegressor(n_estimators=200,max_depth=5,
                             random_state=24,n_jobs=-1)
model3.fit(X_train,Y_train)
y3_pred=model3.predict(X_test)
result3=test_data[['Date','Ticker','target']].copy()
result3['prediction']=y3_pred
daily_ic3=result3.groupby('Date').apply(lambda x: x['prediction'].corr(
    x['target'],method='spearman'
))
print(daily_ic3.describe())