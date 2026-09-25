# French Equity Portfolio Optimization & Risk Analysis

## Overview

This project studies portfolio construction and risk management using a selection of French equities listed on Euronext Paris.

The analysis uses historical market data to:

* calculate logarithmic returns;
* estimate the annualized covariance matrix;
* construct the Minimum Variance Portfolio (MVP);
* derive the efficient frontier;
* identify the portfolio with the highest historical Sharpe ratio;
* estimate Value at Risk (VaR);
* estimate Expected Shortfall (ES).

The project is intended as a quantitative finance / portfolio management exercise.

---

## Assets

The portfolio contains a selection of large French companies from different sectors, including:

| Ticker     | Company            |
| ---------- | ------------------ |
| MC.PA      | LVMH               |
| OR.PA      | L'Oréal            |
| BNP.PA     | BNP Paribas        |
| GLE.PA     | Société Générale   |
| AXA ticker | AXA                |
| ACA.PA     | Crédit Agricole    |
| TTE.PA     | TotalEnergies      |
| SAN.PA     | Sanofi             |
| SU.PA      | Schneider Electric |
| AIR.PA     | Airbus             |

Historical market data is retrieved using `yfinance`.

---

## Methodology

### 1. Historical prices

Daily adjusted closing prices are downloaded from Yahoo Finance.

```python
df = yf.download(
    french_company,
    start="2015-01-01",
    auto_adjust=True
)["Close"]
```

---

### 2. Log returns

Daily logarithmic returns are calculated as:

$$
r_t = \ln(P_t) - \ln(P_{t-1})
$$

```python
ret_matrix = np.log(df).diff().dropna()
```

---

### 3. Annualized covariance matrix

The daily covariance matrix is annualized using 252 trading days:

$$
\Sigma_{annual} = 252 \times \Sigma_{daily}
$$

```python
cov_matrix = ret_matrix.cov() * 252
```

---

### 4. Minimum Variance Portfolio

The unconstrained minimum variance portfolio is calculated under the fully-invested constraint:

$$
\min_w w^T\Sigma w
$$

subject to:

$$
\sum_i w_i = 1
$$

The analytical solution is:

$$
w_{MVP} =
\frac{\Sigma^{-1}\mathbf{1}}
{\mathbf{1}^T\Sigma^{-1}\mathbf{1}}
$$

---

### 5. Efficient Frontier

For a target portfolio return \(R\), the minimum portfolio variance can be expressed using:

$$
A = \mu^T\Sigma^{-1}\mu
$$

$$
B = \mathbf{1}^T\Sigma^{-1}\mu
$$

$$
C = \mathbf{1}^T\Sigma^{-1}\mathbf{1}
$$

$$
D = AC-B^2
$$

and:

$$
\sigma^2(R) = \frac{CR^2 - 2BR + A}{D}
$$

The resulting portfolios are plotted to visualize the efficient frontier.

---

### 6. Sharpe Ratio

The historical Sharpe ratio is calculated using a risk-free rate of 2%:

$$
Sharpe =
\frac{R_p-R_f}{\sigma_p}
$$

```python
rf = 0.02
sharpe_ratio = (returns - rf) / volatility
```

---

### 7. Value at Risk

A parametric Gaussian VaR is estimated at the 95% confidence level.

The calculation assumes normally distributed portfolio returns.

---

### 8. Expected Shortfall

Expected Shortfall is estimated using the Gaussian distribution and represents the expected loss beyond the VaR threshold.

---

## Results

The project produces an efficient frontier together with:

* Minimum Variance Portfolio
* Maximum Historical Sharpe Ratio Portfolio

Example output:

![Efficient Frontier](figures/efficient_frontier.png)

---

## Risk Analysis

The project also estimates:

* Annualized portfolio return
* Annualized volatility
* 95% Value at Risk
* 95% Expected Shortfall

These measures provide a basic view of the portfolio's historical risk profile.

---

## Project Structure

```text
french-equity-portfolio/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── portfolio_analysis.py
│
├── notebooks/
│   └── portfolio_analysis.ipynb
│
├── figures/
│   └── efficient_frontier.png
│
└── data/
    └── README.md
```

---

## Limitations

This project is intended for educational and research purposes.

Several assumptions simplify the analysis:

* historical returns are used as estimates of future returns;
* the covariance matrix is estimated from historical data;
* returns are assumed to follow a Gaussian distribution for the parametric VaR and Expected Shortfall calculations;
* transaction costs are ignored;
* taxes and liquidity constraints are ignored;
* short selling constraints are not explicitly imposed;
* the analysis does not account for portfolio rebalancing over time.

Historical performance should therefore not be interpreted as a prediction of future performance.

---

## Data Source

Market data is obtained from Yahoo Finance through the `yfinance` Python library.

Data availability and historical ticker information may change over time.

---

## Technologies

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib
* Seaborn
* yfinance
* Jupyter Notebook

---

## Disclaimer

This project is for educational purposes only and does not constitute investment advice.
