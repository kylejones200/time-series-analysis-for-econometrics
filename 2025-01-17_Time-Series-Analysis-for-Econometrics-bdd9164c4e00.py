# Description: Short example for Time Series Analysis for Econometrics.



from arch import arch_model
from data_io import read_csv
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



def analyze_stationarity(series, name="Series"):
    """Comprehensive stationarity analysis using ADF test"""
    adf_result = adfuller(series, autolag='AIC')
    logger.info(f"Stationarity Analysis for {name}")
    logger.info(f'ADF Statistic: {adf_result[0]:.4f}')
    logger.info(f'p-value: {adf_result[1]:.4f}')
    logger.info('Critical values:')
    for key, value in adf_result[4].items():
        logger.info(f'\t{key}: {value:.4f}')
    return adf_result[1] < 0.05
def test_cointegration(y1, y2):
    """Test for cointegration between two time series"""
    score, pvalue, _ = coint(y1, y2)
    logger.info("Cointegration Test Results")
    logger.info(f'Test Statistic: {score:.4f}')
    logger.info(f'p-value: {pvalue:.4f}')
    return pvalue < 0.05

class EconometricARIMA:
    def __init__(self, data, order=(1,1,1), seasonal_order=(0,0,0,0)):
        self.data = data
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def fit(self):
        """Fit SARIMA model with automatic differencing"""
        self.model = sm.tsa.SARIMAX(self.data,
                                    order=self.order,
                                    seasonal_order=self.seasonal_order)
        self.results = self.model.fit()
        return self.results
    def diagnostic_plots(self):
        """Generate diagnostic plots for model evaluation"""
        self.results.plot_diagnostics(figsize=(15, 12))
        plt.tight_layout()
        plt.show()
    def forecast(self, steps=10, alpha=0.05):
        """Generate forecasts with confidence intervals"""
        forecast = self.results.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)
        return mean_forecast, conf_int

class EconometricVAR:
    def __init__(self, data, maxlags=None):
        self.data = data
        self.maxlags = maxlags
        self.model = None
        self.results = None

    def select_order(self):
        """Select optimal lag order using information criteria"""
        model = sm.tsa.VAR(self.data)
        return model.select_order(maxlags=self.maxlags)
    def fit(self, lags=None):
        """Fit VAR model"""
        if lags is None:
            lags = self.select_order().aic
        self.model = sm.tsa.VAR(self.data)
        self.results = self.model.fit(lags)
        return self.results
    def granger_causality(self, caused, causing, signif=0.05):
        """Test for Granger causality"""
        test_result = self.results.test_causality(caused, causing, kind='f')
        return {
            'test_statistic': test_result.test_statistic,
            'p_value': test_result.pvalue,
            'significant': test_result.pvalue < signif
        }


class VolatilityAnalysis:
    def __init__(self, returns):
        self.returns = returns
        self.model = None
        self.results = None
    def fit_garch(self, p=1, q=1):
        """Fit GARCH(p,q) model"""
        self.model = arch_model(self.returns, vol='Garch', p=p, q=q)
        self.results = self.model.fit()
        return self.results
    def forecast_volatility(self, horizon=10):
        """Forecast volatility"""
        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance.values[-1]
    def analyze_volatility_clustering(self):
        """Analyze volatility clustering"""
        squared_returns = self.returns**2
        acf = sm.tsa.acf(squared_returns, nlags=20)
        return acf

class ErrorCorrectionModel:
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.results = None

    def fit(self):
        """Fit Error Correction Model"""
        coint_reg = sm.OLS(self.y, sm.add_constant(self.x)).fit()
        residuals = coint_reg.resid
        dy = np.diff(self.y)
        dx = np.diff(self.x)
        res_lag = residuals[:-1]
        X = sm.add_constant(np.column_stack((dx, res_lag)))
        ecm = sm.OLS(dy, X).fit()
        self.results = ecm
        return self.results

class PanelAnalysis:
    def __init__(self, data, entity_col, time_col):
        self.data = data
        self.entity_col = entity_col
        self.time_col = time_col

    def fixed_effects(self, y_col, x_cols):
        """Estimate fixed effects model"""
        model = sm.PanelOLS.from_formula(
            f"{y_col} ~ {'+'.join(x_cols)} + EntityEffects",
            data=self.data.set_index([self.entity_col, self.time_col])
        )
        return model.fit()
    def random_effects(self, y_col, x_cols):
        """Estimate random effects model"""
        model = sm.RandomEffects.from_formula(
            f"{y_col} ~ {'+'.join(x_cols)}",
            data=self.data.set_index([self.entity_col, self.time_col])
        )
        return model.fit()

def analyze_economic_indicators():
    gdp = read_csv('gdp_data.csv', parse_dates=['date'], index_col='date')
    inflation = read_csv('inflation_data.csv', parse_dates=['date'], index_col='date')


    gdp_stationary = analyze_stationarity(gdp['value'], 'GDP')
    inf_stationary = analyze_stationarity(inflation['value'], 'Inflation')
    cointegrated = test_cointegration(gdp['value'], inflation['value'])
    if gdp_stationary and inf_stationary:
        model = EconometricVAR(pd.concat([gdp, inflation], axis=1))
        results = model.fit()
        granger_results = model.granger_causality('gdp', 'inflation')
    else:
        ecm = ErrorCorrectionModel(gdp['value'], inflation['value'])
        results = ecm.fit()
    vol_analysis = VolatilityAnalysis(gdp.pct_change().dropna())
    garch_results = vol_analysis.fit_garch()
    return results, granger_results, garch_results

def detect_structural_breaks(series, max_breaks=5):
    """Detect structural breaks using Bai-Perron test"""
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    scores, pvals = breaks_cusumolsresid(series)
    return scores, pvals

def seasonal_adjustment(series):
    """Perform seasonal adjustment using X-13 ARIMA-SEATS"""
    return sm.tsa.x13_arima_analysis(series)

def handle_missing_economic_data(df):
    """Handle missing values in economic time series"""
    df_interp = df.interpolate(method='cubic')
    df_filled = df_interp.ffill()
    return df_filled
