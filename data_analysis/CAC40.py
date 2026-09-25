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

df = yf.download(french_company, start='2015-01-01',auto_adjust=True)['Close']
print(df.head())
ret_matrix=np.log(df).diff().dropna()
cov_matrix=ret_matrix.cov()*252
in_cov=np.linalg.inv(cov_matrix)
print(cov_matrix.head())
num_assets=10
weight=np.random.dirichlet(np.ones(num_assets)).T
w=weight.reshape(-1,1)
var=w.T@cov_matrix@w
print(var)
m1=np.ones(num_assets).reshape(-1,1)
lam=float(1/((m1.T@in_cov@m1)[0,0]))
print(lam)
w_min=lam*(in_cov@m1)
print(w_min)


#build efficient frontier
np.random.seed(42)
R=np.random.normal(0.1,0.08,1000)
mu=(ret_matrix.mean()*252).to_numpy().reshape(-1,1)
print(mu)
A=mu.T@in_cov@mu
B=m1.T@in_cov@mu
C=m1.T@in_cov@m1
D=A*C-B**2
print('----------')
sigma=np.sqrt(((C*R**2-2*B*R+A)/D))

import seaborn as sns
sns.set_theme(style='darkgrid')
plt.figure(figsize=(10,6),dpi=100)
plt.scatter(sigma,R,color='black',s=50)
# plt.show()

# sharpe ratio
rf=0.02
sr=(R-rf)/sigma
df2=pd.DataFrame({"R": R.flatten(), "SR": sr.flatten(),"sigma":sigma.flatten()})
# print(df2.head())
max_sr=df2['SR'].idxmax()
print(max_sr)
best_por=df2.iloc[max_sr]
print(best_por)

min_var=df2['sigma'].idxmin()
mvp=df2.iloc[min_var]

# to visualize mvp & biggest sharpe ratio
plt.scatter(mvp['sigma'],mvp['R'],marker='*',s=150,color='blue',label='MVP')
plt.scatter(best_por['sigma'],best_por['R'],marker='D',label='Max Sharpe Ratio',s=150)
plt.legend()
plt.show()

# Var & ES
port_ret=ret_matrix@w_min
a=0.95
import scipy.stats as stats
z=stats.norm.ppf(1-a)
fai=stats.norm.pdf(z)
mu=port_ret.mean()*252
vol=port_ret.std()*np.sqrt(252)
var=-(mu+z*vol)
ES=-(mu-(vol*fai)/(1-a))
print(var,ES)



