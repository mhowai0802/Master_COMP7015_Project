## AI Stocks — Buy/Sell Decision Helper

This project helps you explore **buy/sell price decisions** and **direction predictions** for a small watchlist of AI-related stocks.

It is written in **Python**, with:

- **`types/`**: domain objects (e.g. `Stock`, `PredictionInput`, `PredictionOutput`).
- **`api/`**: data access, currently `yfinance` for price history and placeholders for sentiment/fundamental data.
- **`ml/`**: baseline model **plus** Lab-style models:
  - `baseline_model.py`: moving-average baseline.
  - `lab2_mlp_model.py`: Lab 2–style MLP on tabular features (technical, sentiment, fundamentals).
  - `lab5_transformer_model.py`: Lab 5–style Transformer encoder for price sequences.
- **`frontend/`**: a simple Streamlit app to interact with the system.

### Quick Start

**For detailed setup and running instructions, see [HOW_TO_RUN.md](HOW_TO_RUN.md)**

### 1. Setup

In a terminal:

```bash
cd /Users/waiwai/Desktop/Github/Master_COMP7015_Project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Run the frontend

```bash
cd /Users/waiwai/Desktop/Github/Master_COMP7015_Project
streamlit run frontend/app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`) in your browser.

**Note:** The baseline model works immediately. For MLP and Transformer predictions, train the models first (see `HOW_TO_RUN.md`).

### 3. Next steps (how to extend using your course materials)

- **Train the Lab 2 MLP model**:
  - Use the MLP patterns from `COMP7015_Lab2.ipynb`.
  - Build a dataset of feature vectors (see `_build_tabular_features` in `lab2_mlp_model.py`)
    and future N-day returns as labels.
  - Train a regression model and save weights to `models/stock_mlp.pth` so the frontend
    can show its predictions.
- **Train the Lab 5 Transformer-style model**:
  - Use the attention / Transformer patterns from
    `COMP7015 - Lab 5 - Attention and Transformer.ipynb`.
  - Build sequences of daily features (see `_build_sequence_features` in
    `lab5_transformer_model.py`) and predict future N-day returns.
  - Save weights to `models/stock_transformer.pth` for the frontend to load.
- **Add sentiment analysis** for X posts:
  - Implement a real `fetch_sentiment_score` in `api/price_data.py` (or a new module).
  - Train or reuse a sentiment classifier, and feed its score into your prediction model.
- **Add fundamental analysis features**:
  - Extend `fetch_fundamental_snapshot` to pull valuation and financial metrics from Yahoo Finance.
  - Convert them into numeric features and use them in your ML model.

This gives you a clean structure to experiment with different data sources and models while keeping
your **Python objects**, **API calls**, **ML logic**, and **frontend** well separated.


