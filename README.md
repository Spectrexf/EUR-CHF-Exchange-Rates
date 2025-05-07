What the Code Does:

Data Collection: Fetches historical EUR/CHF data (2010–2024) using Yahoo Finance.

SNB Intervention Tagging: Flags key dates (e.g., the 2015 peg removal and 2020 COVID-19 interventions).

Preprocessing: Scales data and creates time-series windows for LSTM training.

Enhanced LSTM Model:

Architecture: 3 LSTM layers with dropout regularization (128, 64, 32 units).

Metrics: Evaluates performance using MAE (Mean Absolute Error) and R² scores.

Interactive Visualization: Compares actual vs. predicted values with Plotly.

Technical Highlights:

✅ Hybrid Inputs: Combines price data and SNB intervention flags as features.

✅ Robust Regularization: Dropout layers (30%, 20%) to prevent overfitting.

✅ Explainability: Clear visualizations to track model accuracy during volatility.

Business Implications:

Central bank actions are black swan events for forex markets. This model demonstrates how ML can adapt to regulatory shocks.

Potential applications: Risk management, algorithmic trading strategies, or macroeconomic research.
