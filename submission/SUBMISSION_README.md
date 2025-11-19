# Submission Package for FSC 8/F Lab Environment

## Contents

This submission package contains:

1. **Project Report**: `Project_Report.md` (to be converted to PDF)
2. **Source Code**: All Python source files required to run the project

## Setup Instructions for FSC 8/F Lab

### 1. Extract the submission package

Extract the zip file to a directory (e.g., `~/ai_stocks_project/`)

### 2. Create a virtual environment (recommended)

```bash
cd ~/ai_stocks_project
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The lab environment may already have some packages installed. If you encounter dependency conflicts, try installing packages individually or use `pip install --upgrade <package>`.

### 4. Download NLTK data (required for text preprocessing)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 5. Configure API keys (optional)

If you have a NewsAPI key for sentiment analysis:

1. Create or edit `config/api_keys.py`
2. Add: `NEWSAPI_KEY = "your_api_key_here"`

**Note**: The system works without NewsAPI - it will use VADER sentiment analysis as a fallback. However, having a NewsAPI key enables news headline fetching and deeper sentiment analysis.

### 6. Train models (optional)

Pre-trained models are included in the `models/` directory:
- `stock_mlp.pth`: MLP model weights
- `stock_transformer.pth`: Transformer model weights
- `sentiment_lstm.pth`: LSTM sentiment model weights

To retrain models:

```bash
# Train MLP model
python -m ml.train_mlp_model

# Train Transformer model
python -m ml.train_transformer_model

# Train sentiment model
python -m ml.train_sentiment_model
```

### 7. Run the application

```bash
streamlit run frontend/app.py
```

The application will open in your browser at `http://localhost:8501`

## Project Structure

```
AI_Stocks/
├── api/                    # Data access layer
│   ├── price_data.py      # Price and fundamental data fetching
│   └── news_sentiment.py  # News and sentiment analysis
├── config/                # Configuration files
│   └── api_keys.py        # API keys (create this if needed)
├── data/                  # Data files (sentiment dataset, etc.)
├── domain/                # Domain objects and business logic
│   ├── stocks.py
│   ├── predictions.py
│   ├── timing.py
│   └── configs.py
├── frontend/              # Streamlit web application
│   └── app.py
├── ml/                    # Machine learning models
│   ├── baseline_model.py
│   ├── lab2_mlp_model.py
│   ├── lab5_transformer_model.py
│   ├── sentiment_lstm_model.py
│   ├── sentiment_bert_model.py
│   ├── train_mlp_model.py
│   ├── train_transformer_model.py
│   ├── train_sentiment_model.py
│   └── ...
├── models/                # Trained model weights
│   ├── stock_mlp.pth
│   ├── stock_transformer.pth
│   └── sentiment_lstm.pth
├── cache/                 # Cached data (created automatically)
├── requirements.txt       # Python dependencies
├── Project_Report.md      # Project report (convert to PDF)
└── SUBMISSION_README.md   # This file
```

## Key Features

1. **Multiple ML Models**: Baseline, MLP (Lab 2-style), and Transformer (Lab 5-style) models for stock direction prediction
2. **Sentiment Analysis**: Deep learning models (LSTM/BERT) for financial sentiment from news headlines
3. **Technical & Fundamental Analysis**: Price indicators, moving averages, P/E ratios, market cap, etc.
4. **Scenario Simulation**: Monte Carlo path simulation for risk assessment
5. **Intraday Analysis**: Optimal timing recommendations based on historical volatility patterns
6. **Interactive Web Interface**: Streamlit app for easy interaction

## Watchlist Stocks

The system supports 10 AI-related stocks:
- AAPL (Apple)
- MSFT (Microsoft)
- NVDA (NVIDIA)
- GOOGL (Alphabet/Google)
- AMZN (Amazon)
- META (Meta/Facebook)
- TSLA (Tesla)
- AVGO (Broadcom)
- TSM (TSMC)
- SMCI (Super Micro Computer)

## Troubleshooting

### Issue: Import errors

**Solution**: Ensure all dependencies are installed and the project root is on Python path. The application adds the project root to `sys.path` automatically.

### Issue: Model files not found

**Solution**: Models should be in the `models/` directory. If missing, train them using the commands in step 6.

### Issue: API rate limits (Yahoo Finance, NewsAPI)

**Solution**: The system uses local caching to minimize API calls. Cache files are stored in `cache/` directory. Delete cache files if you want fresh data.

### Issue: NLTK data not found

**Solution**: Run the NLTK download command in step 4.

### Issue: CUDA/GPU errors

**Solution**: Models default to CPU inference. The code automatically handles GPU availability. If you see CUDA errors, the system will fall back to CPU.

## Testing

To verify the installation:

```bash
# Test imports
python -c "from ml.lab2_mlp_model import StockMLP; from ml.lab5_transformer_model import StockTransformer; print('Imports successful')"

# Test data fetching
python -c "from api.price_data import fetch_price_history; from domain.stocks import Stock; s = Stock('Apple', 'AAPL'); series = fetch_price_history(s, lookback_days=30); print(f'Fetched {len(series.prices)} price points')"
```

## Notes

- The system requires internet connection for data fetching (Yahoo Finance, NewsAPI)
- First run may be slower due to data fetching and caching
- Cache files can be large; clean `cache/` directory periodically if disk space is limited
- All models support CPU inference; GPU is optional

## Contact

For questions about the submission, please refer to the Project Report or contact the project team.

---

**Environment**: Tested on Python 3.8+, Linux/macOS/Windows compatible

