# Time Series Analysis for Econometrics

This project demonstrates time series analysis methods for econometric modeling.

## Business context

Economic data has properties that make standard time series methods either wrong or incomplete. GDP, inflation, interest rates, and asset prices are rarely stationary. They trend, cointegrate, exhibit volatility clustering, and break structurally when policy changes. Understanding these properties — and the methods designed to handle them — is what separates rigorous economic analysis from curve fitting.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Econometric analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize analysis parameters and output settings.

## Econometric Methods

- ADF Test: Tests for unit roots and stationarity
- Cointegration: Tests for long-run equilibrium relationships
- VAR Model: Vector Autoregression for multivariate time series

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).