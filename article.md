# Time Series Analysis for Econometrics

*Exploring the intersection of economic theory and time series analysis*

---

Economic data has properties that make standard time series methods either wrong or incomplete. GDP, inflation, interest rates, and asset prices are rarely stationary. They trend, cointegrate, exhibit volatility clustering, and break structurally when policy changes. Understanding these properties — and the methods designed to handle them — is what separates rigorous economic analysis from curve fitting.

## Testing for Unit Roots and Cointegration

Most economic time series are non-stationary: their mean and variance change over time. Running OLS on non-stationary series produces spurious regressions — high R² and low p-values that reflect nothing but shared trends.

The Augmented Dickey-Fuller (ADF) test checks for a unit root:

```python
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

def adf_test(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"{name}: ADF={result[0]:.4f}, p={result[1]:.4f}, stationary={result[1] < 0.05}")
    return result[1] < 0.05

is_stationary = adf_test(df['gdp_growth'], 'GDP Growth')
```

If two non-stationary series move together over time — their linear combination is stationary — they are cointegrated. Cointegration is economically meaningful: it implies a long-run equilibrium relationship even though the individual series wander.

```python
# Engle-Granger cointegration test
score, pvalue, _ = coint(df['consumption'], df['income'])
print(f"Cointegration p-value: {pvalue:.4f}")
```

If series are cointegrated, do not difference them before modeling — that throws away the long-run relationship. Use an Error Correction Model instead (see below).

## ARIMA Models with Economic Data

ARIMA(p, d, q) models handle non-stationarity through differencing (the I term) and capture autocorrelation through AR and MA terms. For economic data, the d parameter is usually 1 — most macro series are integrated of order 1.

```python
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

model = ARIMA(df['gdp_growth'], order=(2, 1, 1))
result = model.fit()
print(result.summary())

forecast = result.get_forecast(steps=8)
print(forecast.conf_int())
```

For seasonal economic data (retail sales, employment, housing starts), use SARIMA to capture the seasonal pattern explicitly rather than relying on seasonal dummies:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['retail_sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit(disp=False)
```

Use AIC or BIC to select lag orders. Do not select orders by minimizing in-sample fit — that overfits.

## Vector Autoregression (VAR) Models

VAR treats multiple economic variables as mutually dependent. Each variable is regressed on its own lags and the lags of every other variable. This captures feedback loops: monetary policy affects output, which affects inflation, which feeds back to policy.

```python
from statsmodels.tsa.api import VAR

var_data = df[['gdp_growth', 'inflation', 'fed_funds_rate']].dropna()
model = VAR(var_data)
result = model.fit(maxlags=8, ic='aic')
print(result.summary())

# Granger causality test within the VAR
result.test_causality('gdp_growth', ['fed_funds_rate'], kind='f')
```

The impulse response function traces how a shock to one variable propagates through the system:

```python
irf = result.irf(periods=20)
irf.plot(orth=True, figsize=(12, 8))
```

The ordering of variables in Cholesky orthogonalization matters — variables listed first are assumed to be causally prior. This should reflect economic theory (e.g., policy variables often go last, as they respond to economic conditions contemporaneously).

## GARCH Models for Volatility Analysis

Financial and energy price series exhibit volatility clustering: large moves tend to cluster together. Standard ARIMA assumes constant variance — GARCH models the variance as a function of past squared errors and past variance.

```python
from arch import arch_model

returns = df['price'].pct_change().dropna() * 100

model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
result = model.fit(disp='off')
print(result.summary())

# Conditional volatility
vol = result.conditional_volatility
```

GARCH(1,1) is the workhorse. The persistence parameter α + β measures how long volatility shocks persist — values near 1.0 indicate long-memory volatility, common in equity and commodity markets. For asymmetric responses (downside shocks increase volatility more than upside), use GJR-GARCH or EGARCH.

## Error Correction Models (ECM)

When two or more series are cointegrated, an ECM captures both the long-run equilibrium relationship and short-run dynamics:

```python
import statsmodels.api as sm
import numpy as np

# Step 1: Estimate long-run relationship
long_run = sm.OLS(df['consumption'], sm.add_constant(df['income'])).fit()
residuals = long_run.resid

# Step 2: ECM — short-run changes + error correction term
delta_c = df['consumption'].diff()
delta_y = df['income'].diff()
ecm_data = pd.DataFrame({
    'delta_c': delta_c,
    'delta_y': delta_y,
    'ec_term': residuals.shift(1)
}).dropna()

ecm = sm.OLS(ecm_data['delta_c'], sm.add_constant(ecm_data[['delta_y', 'ec_term']])).fit()
print(ecm.summary())
```

The coefficient on the error correction term (`ec_term`) measures the speed of adjustment back to equilibrium. A coefficient of -0.3 means 30% of the deviation from equilibrium is corrected each period. It should be negative and statistically significant for the ECM to make sense.

## Panel Data Analysis

Panel data combines cross-sectional units (countries, firms, states) with time series. It controls for unobserved heterogeneity that would bias a pure time series or cross-sectional analysis.

```python
from linearmodels.panel import PanelOLS, RandomEffects

df_panel = df.set_index(['entity_id', 'year'])

# Fixed effects — controls for all time-invariant entity characteristics
fe_model = PanelOLS(
    df_panel['outcome'],
    df_panel[['policy_var', 'control_1', 'control_2']],
    entity_effects=True,
    time_effects=True
)
fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)
print(fe_result.summary)
```

Use the Hausman test to choose between fixed and random effects. Fixed effects is the safe default when you suspect entity-specific unobservables are correlated with your regressors.

## Structural Breaks

Economic relationships break. The Phillips curve, the yield curve spread as a recession predictor, purchasing power parity — all have periods where the historical relationship stops holding. Ignoring structural breaks produces biased estimates and poor forecasts.

```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.recursive_ls import RecursiveLS

# CUSUM test for parameter stability
rls = RecursiveLS(df['y'], sm.add_constant(df[['x1', 'x2']])).fit()
rls.plot_cusum()

# Chow test for a known break date
break_date = '2008-01-01'
df['post_break'] = (df.index >= break_date).astype(int)
df['x1_post'] = df['x1'] * df['post_break']
chow_model = sm.OLS(df['y'], sm.add_constant(df[['x1', 'post_break', 'x1_post']])).fit()
print(chow_model.summary())
```

If the break date is unknown, the Bai-Perron test searches for multiple structural breaks simultaneously. The `ruptures` package provides a Python implementation.

## Seasonal Adjustment

Most economic data is seasonally adjusted before publication (BLS, BEA), but raw data requires explicit treatment. STL decomposition is flexible and handles changing seasonal strength:

```python
from statsmodels.tsa.seasonal import STL

stl = STL(df['retail_sales'], period=12, robust=True)
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

result.plot()
```

The Census Bureau's X-13ARIMA-SEATS is the official standard for government statistics and is available through `statsmodels` if you have the X-13 binary installed. For most analytical work, STL is sufficient.

## Handling Missing Data

Economic time series have gaps from reporting lags, revisions, and methodology changes. How you handle them matters:

```python
# Cubic interpolation for short gaps (< 3 periods)
df['gdp_interpolated'] = df['gdp'].interpolate(method='cubic', limit=3)

# Forward fill for administrative gaps (e.g., holidays in daily data)
df['price_filled'] = df['price'].ffill(limit=2)

# For structural gaps, document and treat as missing in the model
# rather than imputing
```

Do not blindly impute long gaps. A gap of 6+ months in monthly data often reflects a real break — imputing it obscures the discontinuity and biases any analysis that spans it.

## Key Takeaways

- Test for unit roots before modeling. Regressing non-stationary series on each other produces spurious results.
- Cointegrated series share a long-run equilibrium — use Error Correction Models, not differenced VARs.
- GARCH models are necessary for financial series where variance clusters over time.
- Panel data controls for entity heterogeneity; always cluster standard errors by entity.
- Structural breaks are the norm in economic data, not the exception — test for them explicitly rather than assuming a stable relationship.
