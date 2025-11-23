# How to Run the AI Stocks Project

This guide will walk you through setting up and running the entire AI Stocks project.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Internet connection (for downloading stock data and news)

## Step 1: Setup Environment

### 1.1 Navigate to Project Directory

```bash
cd /Users/waiwai/Desktop/Github/Master_COMP7015_Project
```

### 1.2 Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
```

### 1.3 Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter any issues, you may need to install additional dependencies:
```bash
pip install accelerate  # Required for BERT training (optional)
```

## Step 2: Configure API Keys (Optional)

The project uses NewsAPI for fetching news headlines. The API key is already configured in `config/api_keys.py`, but you can update it if needed:

```python
# config/api_keys.py
NEWSAPI_KEY = "your_api_key_here"
```

**Note:** The project will still work without NewsAPI - it will just skip news fetching.

## Step 3: Run the Application

### Option A: Quick Start (Baseline Model Only)

The baseline model works immediately without training:

```bash
streamlit run frontend/app.py
```

Then open your browser to the URL shown in the terminal (usually `http://localhost:8501`).

### Option B: Full Setup (With Trained Models)

For the best experience with MLP and Transformer predictions, train the models first:

#### 3.1 Train MLP Model

```bash
python -m model_code.train_mlp_model
```

This will:
- Download historical stock data
- Build training dataset
- Train the MLP model
- Save weights to `saved_models/stock_mlp.pth`

#### 3.2 Train Transformer Model

```bash
python -m model_code.train_transformer_model
```

This will:
- Download historical stock data
- Build sequence dataset
- Train the Transformer model
- Save weights to `saved_models/stock_transformer.pth`

#### 3.3 Train Sentiment Models (Optional)

**LSTM Sentiment Model:**
```bash
python -m model_code.train_sentiment_model --epochs 50 --batch-size 64
```

**BERT Sentiment Model (requires accelerate):**
```bash
pip install accelerate
python -m model_code.train_bert_sentiment --epochs 10 --batch-size 16
```

#### 3.4 Run the Frontend

After training (or even without training), run:

```bash
streamlit run frontend/app.py
```

## Step 4: Using the Application

1. **Select a Stock**: Choose from the dropdown in the sidebar (e.g., Apple, Microsoft, NVIDIA, etc.)

2. **Click "🚀 開始分析"**: This will:
   - Fetch price data from Yahoo Finance
   - Fetch news headlines (if API key is configured)
   - Fetch fundamental data
   - Run predictions using available models

3. **View Results**: The app displays:
   - **Model Predictions Tab**: Shows predictions from Baseline, MLP, and Transformer models
   - **Scenario Simulation Tab**: Monte Carlo simulation of future price paths
   - **News & Fundamentals Tab**: Recent news and financial metrics
   - **Best Monitoring Times Tab**: Optimal times to monitor the stock based on intraday volatility

4. **Retrain Models**: Click "🔄 重新訓練模型" in the sidebar to retrain MLP and Transformer models

## Step 5: Evaluate All Models (Optional)

To get comprehensive evaluation results for all models:

```bash
python scripts/get_all_model_results.py
```

This will evaluate:
- Stock prediction models (MLP, Transformer, Baseline)
- Sentiment analysis models (LSTM, BERT)

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named '_ctypes'`

This error occurs on macOS when Python was installed without proper system dependencies.

**Solution:**
1. Install libffi via Homebrew:
   ```bash
   brew install libffi
   ```

2. If using pyenv, reinstall Python with libffi support:
   ```bash
   export LDFLAGS="-L/opt/homebrew/opt/libffi/lib"
   export CPPFLAGS="-I/opt/homebrew/opt/libffi/include"
   export PKG_CONFIG_PATH="/opt/homebrew/opt/libffi/lib/pkgconfig"
   pyenv install 3.9.18  # or newer version
   pyenv local 3.9.18
   ```

3. Then recreate your virtual environment:
   ```bash
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

**Note:** Python 3.9+ handles libffi dependencies better than older versions.

### Issue: Models not found / Predictions show "尚未有訓練好的模型權重"

**Solution:** Train the models first using Step 3.1 and 3.2 above.

### Issue: Import errors

**Solution:** Make sure you're in the project root directory and the virtual environment is activated.

### Issue: News API errors

**Solution:** The app will continue to work without news. Check `config/api_keys.py` if you want to enable news features.

### Issue: yfinance data fetching fails

**Solution:** 
- Check your internet connection
- yfinance may have rate limits - wait a few minutes and try again
- Some stocks may not be available on Yahoo Finance

### Issue: CUDA/GPU errors

**Solution:** The project will automatically use CPU if GPU is not available. For GPU support, ensure PyTorch is installed with CUDA support.

## Project Structure

- `frontend/app.py` - Main Streamlit application
- `model_code/` - All ML models and training scripts
- `api/` - Data fetching (price data, news, fundamentals)
- `domain/` - Domain objects and business logic
- `cache/` - Cached data files (auto-generated)
- `config/` - Configuration files including API keys

## Quick Reference Commands

```bash
# Setup
cd /Users/waiwai/Desktop/Github/Master_COMP7015_Project
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Train models
python -m model_code.train_mlp_model
python -m model_code.train_transformer_model

# Run app
streamlit run frontend/app.py

# Evaluate all models
python scripts/get_all_model_results.py
```

## Notes

- The baseline model works immediately without training
- MLP and Transformer models need to be trained before they can make predictions
- Training may take several minutes depending on your hardware
- Data is cached in the `cache/` directory to speed up subsequent runs
- The app fetches live data from Yahoo Finance, so you need an internet connection

## Support

For more details, see:
- `README.md` - Project overview
- `docs/Model_Technical_Documentation.md` - Technical documentation
- `results/MODEL_RESULTS_SUMMARY.md` - Model performance results

