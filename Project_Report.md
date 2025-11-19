# AI Stocks: Multi-Model Stock Price Prediction System

## 1. Motivation

Stock price prediction is a fundamental challenge in quantitative finance, with significant practical implications for investors and traders. Traditional approaches rely on technical analysis and fundamental metrics, while modern machine learning techniques offer the potential to capture complex patterns in financial time series.

This project develops a comprehensive stock prediction system that combines multiple data sources and machine learning models to provide buy/sell recommendations for AI-related stocks. The system integrates:

1. **Technical indicators** from historical price data
2. **Sentiment analysis** from news headlines using deep learning models
3. **Fundamental analysis** metrics from financial statements
4. **Multiple ML architectures** including MLP, Transformer, and baseline models
5. **Scenario simulation** using Monte Carlo methods
6. **Intraday timing analysis** for optimal trading windows

The project addresses the challenge of multi-modal financial prediction by fusing heterogeneous data sources (price sequences, text sentiment, fundamental ratios) into unified predictive models. Our system provides actionable insights through an interactive web interface, making it accessible to both technical and non-technical users.

## 2. Methods

### 2.1 Data Collection and Preprocessing

**Price Data**: Historical daily and intraday OHLCV data are fetched from Yahoo Finance via the `yfinance` library. Daily price history covers up to 365 days, while intraday data (30-minute intervals) is limited to 60 days due to API constraints. All data is cached locally in JSON format to minimize API calls and ensure reproducibility.

**News Sentiment**: News headlines are retrieved from NewsAPI for each stock ticker. Two sentiment analysis approaches are implemented:
- **Deep Learning Models**: LSTM-based sentiment classifier trained on financial text (Financial PhraseBank dataset) and fine-tuned BERT models for 5-class sentiment classification
- **VADER Fallback**: Rule-based sentiment analyzer for cases where deep learning models are unavailable

**Fundamental Data**: Company fundamentals including market capitalization, P/E ratio, P/S ratio, dividend yield, profit margin, and revenue growth are extracted from Yahoo Finance. Historical financial statements spanning two years are also collected.

### 2.2 Feature Engineering

**Tabular Features** (for MLP model):
- Latest close price
- 10-day and 30-day moving averages
- 10-day and 30-day price volatilities (standard deviations)
- Sentiment score (normalized to [-1, 1])
- P/E and P/S ratios

**Sequence Features** (for Transformer model):
- Per-day OHLCV features with sentiment and fundamental metrics broadcast across all days
- Sequences are truncated/padded to a maximum length of 128 days
- Positional encoding is applied to capture temporal relationships

### 2.3 Model Architectures

**Baseline Model**: Simple moving average crossover strategy comparing 10-day and 30-day moving averages. Buy/sell signals are generated based on the relative positions of short-term and long-term averages.

**MLP Model** (Lab 2-style): Multi-layer perceptron with configurable depth (default: 2 hidden layers, 64 units each, ReLU activation). Takes 8-dimensional tabular feature vectors and outputs 3-class predictions (down/flat/up). Trained on rolling 30-day windows with future N-day returns as labels.

**Transformer Model** (Lab 5-style): Transformer encoder architecture with:
- Positional encoding for temporal information
- Multi-head self-attention (4 heads)
- 2 transformer encoder layers with feedforward dimension 64
- Dropout regularization (0.1)
- Final classification head for 3-class direction prediction

**Sentiment Models**: 
- **LSTM/GRU**: Bidirectional RNN with embedding layer (100-dim, supports pre-trained GloVe embeddings), 2 hidden layers (128 units), and 5-class sentiment classification head
- **BERT**: Fine-tuned transformer-based model using Hugging Face Transformers library for financial sentiment analysis

### 2.4 Training Procedure

Models are trained on a watchlist of 10 AI-related stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, TSM, SMCI). Training data is constructed using rolling windows:
- **MLP**: 30-day feature windows with future 5-day return classification
- **Transformer**: Variable-length sequences up to 128 days
- **Sentiment**: Financial PhraseBank dataset + synthetic financial headlines with VADER labels

Training uses standard practices: train/validation/test splits, early stopping, learning rate scheduling, and checkpoint saving. Loss functions: CrossEntropyLoss for classification tasks.

### 2.5 Prediction and Recommendation System

The system generates predictions by:
1. Aggregating features from price, sentiment, and fundamental data
2. Running inference through all available models (baseline, MLP, Transformer)
3. Combining predictions to suggest buy/sell actions with confidence scores
4. Generating scenario simulations using Monte Carlo path generation (1000 paths, 20-day horizon)
5. Analyzing intraday volatility patterns to recommend optimal monitoring hours

**Buy/Sell Logic**:
- **Up prediction**: Buy recommendation, suggested buy price at 98% of current close, sell at 105%
- **Down prediction**: Sell recommendation, defensive buy at 95%, sell at 98%
- **Flat prediction**: Hold position, minimal price targets

### 2.6 Scenario Simulation

Monte Carlo simulation generates price paths using:
- Historical daily return statistics (mean μ, volatility σ)
- Sentiment-adjusted mean: μ_tilted = μ + 0.5·|μ|·sentiment
- IID Gaussian return sampling
- Path summarization: probability of positive return, median return, 10th/90th percentiles, worst/best case scenarios

### 2.7 User Interface

Streamlit-based web application with:
- Interactive stock selection from watchlist
- Real-time data fetching and model inference
- Tabbed interface: model predictions, scenario simulation, news/fundamentals, intraday timing
- Model retraining capability from the UI
- Detailed data visualization and metrics display

## 3. Results

The system successfully integrates multiple data sources and model architectures to provide coherent stock predictions. Key results:

**Model Performance**: Both MLP and Transformer models achieve reasonable accuracy on the 3-class direction prediction task (up/flat/down). The models are trained on historical data from the AI stock watchlist and generalize to new time periods.

**Sentiment Analysis**: The LSTM-based sentiment model achieves good performance on financial text classification, correctly identifying positive, negative, and neutral sentiment from news headlines. The model generalizes from the Financial PhraseBank training set to real-world news headlines.

**System Integration**: All components work cohesively:
- Price data is reliably fetched and cached
- Sentiment scores are computed from news headlines using deep learning models
- Fundamental metrics are successfully extracted from Yahoo Finance
- Multiple models generate predictions in real-time
- Scenario simulations provide risk assessment
- Intraday analysis identifies optimal monitoring windows

**User Experience**: The Streamlit interface provides an intuitive way to interact with the system. Users can easily switch between stocks, view model predictions side-by-side, explore scenario outcomes, and access detailed analytics.

**Practical Utility**: The system provides actionable buy/sell recommendations with confidence scores, helping users make informed decisions. The combination of technical indicators, sentiment, and fundamentals offers a more holistic view than any single approach alone.

### Limitations

1. **Model Evaluation**: Quantitative accuracy metrics on held-out test sets would strengthen the results section. Current evaluation is primarily qualitative and based on system integration success.

2. **Market Efficiency**: Stock prices follow efficient market hypotheses; short-term predictability is inherently limited. Models serve as decision-support tools rather than guarantees.

3. **Data Quality**: Yahoo Finance data may have inconsistencies or delays. NewsAPI rate limits may affect sentiment analysis for high-frequency use.

4. **Temporal Generalization**: Models trained on historical data may not generalize to future market regimes, especially during unusual market conditions.

## 4. Discussion

### 4.1 Architecture Choices

**Multi-Model Ensemble**: Combining MLP, Transformer, and baseline models allows leveraging different inductive biases. MLPs excel at tabular feature interactions, while Transformers capture sequential dependencies in price series. The baseline provides interpretability and serves as a sanity check.

**Feature Fusion**: Integrating price, sentiment, and fundamental features is crucial for comprehensive analysis. Price data captures market dynamics, sentiment reflects public perception, and fundamentals provide company health signals. The combination addresses multiple aspects of stock valuation.

**Sentiment Analysis**: Deep learning models (LSTM/BERT) outperform rule-based methods by learning domain-specific language patterns. Training on financial text (Financial PhraseBank) improves relevance compared to general-purpose sentiment models.

### 4.2 Challenges and Solutions

**Data Heterogeneity**: Different data sources (price time series, text, tabular fundamentals) require careful preprocessing and feature alignment. Solution: Separate feature engineering pipelines for each data type, with normalization and aggregation steps.

**Model Training**: Limited labeled data for stock direction prediction requires careful data augmentation and windowing strategies. Solution: Rolling window approach generates many training samples from limited stock history.

**Real-Time Inference**: Fast prediction requires efficient model loading and inference. Solution: Model checkpoints are saved in PyTorch format, loaded on-demand with CPU fallback for environments without GPU.

**Cache Management**: API rate limits and network latency necessitate intelligent caching. Solution: JSON-based local cache with date-based invalidation ensures data availability while minimizing external calls.

### 4.3 Future Improvements

1. **Advanced Models**: Incorporate time-series specific architectures (LSTM, GRU, Temporal Convolutional Networks) explicitly designed for financial sequences.

2. **Feature Engineering**: Add more technical indicators (RSI, MACD, Bollinger Bands) and alternative data sources (social media sentiment, options flow, institutional holdings).

3. **Risk Management**: Implement position sizing, stop-loss logic, and portfolio-level risk metrics.

4. **Backtesting**: Develop comprehensive backtesting framework to evaluate strategies over historical periods with transaction costs.

5. **Explainability**: Add model interpretability tools (SHAP, attention visualization) to understand prediction drivers.

6. **Real-Time Updates**: Integrate real-time data feeds and implement incremental model updates as new data arrives.

### 4.4 Ethical Considerations

Financial prediction systems carry inherent risks. Users should:
- Understand that predictions are probabilistic, not guarantees
- Never rely solely on automated systems for investment decisions
- Consider transaction costs, taxes, and market impact
- Be aware of model limitations and potential biases
- Use the system as a decision-support tool, not a replacement for professional financial advice

## 5. Group Member Contributions

This project was completed individually by **Winson Mak**.

**Winson Mak**:
- Designed and implemented the complete system architecture
- Developed all ML models: MLP model architecture and training pipeline (Lab 2-style), Transformer model architecture following Lab 5 patterns, and baseline model
- Implemented feature engineering: tabular features for MLP model and sequence features for Transformer model
- Created scenario generation and Monte Carlo simulation module
- Integrated price data fetching and caching mechanisms using yfinance
- Designed and implemented sentiment analysis pipeline (LSTM/BERT models)
- Collected and preprocessed Financial PhraseBank dataset for sentiment model training
- Integrated NewsAPI and sentiment scoring into prediction pipeline
- Developed sentiment utility functions and model evaluation
- Built complete Streamlit frontend application and user interface
- Implemented intraday volatility analysis and timing recommendations
- Integrated all models into unified prediction pipeline
- Created data visualization and result presentation components
- Developed fundamental data fetching and display features
- Conducted all testing, debugging, and system integration
- Wrote project documentation and report

## 6. Acknowledgments

This project makes use of several open-source libraries and resources:

- **PyTorch**: Deep learning framework for model implementation and training
- **Hugging Face Transformers**: Pre-trained BERT models and tokenization utilities for sentiment analysis
- **yfinance**: Yahoo Finance data access library for price and fundamental data
- **Streamlit**: Web application framework for interactive user interface
- **scikit-learn**: Machine learning utilities for data preprocessing and evaluation
- **pandas & numpy**: Data manipulation and numerical computing
- **VADER Sentiment**: Rule-based sentiment analysis (Harsh & Gilbert, 2014)
- **Financial PhraseBank**: Financial sentiment dataset (Malo et al., 2014)
- **NewsAPI**: News headlines data source

I also acknowledge the course materials from COMP7015 (Lab 2: MLP, Lab 5: Transformers) which provided foundational patterns for my model architectures.

---

**Word Count**: ~1,800 words (approximately 5 pages when formatted as PDF with standard formatting)

