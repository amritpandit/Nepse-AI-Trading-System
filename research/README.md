# Research

Research notebooks, strategy experiments and the documented market/sector/stock scoring methodology.

## Structure

```
research/
├── SCORING_METHODOLOGY.md
├── notebooks/
│   ├── README.md
│   ├── 01_idea_to_backtest.ipynb
│   ├── 02_backtest_and_export.ipynb
│   └── 03_integrate_bundle.ipynb
└── README.md
```

## Live research scoring

The dashboard's Sector & Stock Research page uses the documented 100-point BUY/SELL evidence model in [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md). It combines NEPSE market regime, multi-horizon sector relative strength, 60/120/200-session stock momentum, trend structure, RSI/Stochastic, MACD, volume/OBV, price action and lower-weight SMC OHLC proxies.

The NEPSE Index is a benchmark only. It is not a tradable instrument in this app. A BUY trigger is not an order; a SELL trigger is a review/exit alert for an existing long holding. SMC labels are deterministic chart proxies, not verified institutional order flow.

## Workflow

1. **Idea**: Define hypothesis in `01_idea_to_backtest.ipynb`
2. **Experiment**: Execute backtest via API
3. **Export**: Download artifacts in `02_backtest_and_export.ipynb`
4. **Integrate**: Package for production in `03_integrate_bundle.ipynb`

## Outputs

All experimental results are exported to `/exports/`:

- `backtest_report_<id>.json` - Full backtest results
- `equity_curve_<id>.csv` - Time series equity data
- `trades_<id>.csv` - Trade history
- `equity_curve_<id>.png` - Chart visualization

## Prerequisites

- Backend API running (`uvicorn backend.app.main:app`)
- `API_BASE` environment variable set
- Python 3.12 with Jupyter installed
