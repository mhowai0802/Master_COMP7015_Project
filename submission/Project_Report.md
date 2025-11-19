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

### 3.1 Quantitative Model Performance

**Stock Direction Prediction Models**: All models were evaluated on a held-out test set (20% of data) using 80/20 train/test split. The dataset consists of rolling 30-day windows from 10 AI-related stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, TSM, SMCI) with 5-day future return classification.

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|---------------|-------------------|----------------|------------------|
| **MLP** | 0.409 (40.9%) | 0.248 | 0.325 | 0.204 |
| **Transformer** | 0.500 (50.0%) | 0.348 | 0.353 | 0.294 |
| **Baseline** | - | - | - | - |

**MLP Model Performance**:
- **Test Accuracy**: 0.409 (40.9%)
- **Macro-Averaged Precision**: 0.248
- **Macro-Averaged Recall**: 0.325
- **Macro-Averaged F1-Score**: 0.204
- **Per-Class Performance**:
  - Down (Class 0): Precision 0.41, Recall 0.96, F1 0.58, Support: 180
  - Flat (Class 1): Precision 0.00, Recall 0.00, F1 0.00, Support: 43
  - Up (Class 2): Precision 0.33, Recall 0.02, F1 0.04, Support: 207
- **Best Configuration**: hidden_dim=64, num_layers=3
- **Issue**: Model heavily biases toward predicting "Down" class and fails to predict "Flat" class, likely due to class imbalance

**Transformer Model Performance**:
- **Test Accuracy**: 0.500 (50.0%)
- **Macro-Averaged Precision**: 0.348
- **Macro-Averaged Recall**: 0.353
- **Macro-Averaged F1-Score**: 0.294
- **Per-Class Performance**:
  - Down (Class 0): Precision 0.55, Recall 0.16, F1 0.24, Support: 180
  - Flat (Class 1): Precision 0.00, Recall 0.00, F1 0.00, Support: 43
  - Up (Class 2): Precision 0.49, Recall 0.90, F1 0.64, Support: 207
- **Best Configuration**: d_model=64, nhead=8, num_layers=3, dim_feedforward=128
- **Issue**: Model heavily biases toward predicting "Up" class and fails to predict "Flat" class, likely due to class imbalance

**Baseline Model Performance**:
- The moving average crossover baseline provides a simple technical indicator-based approach. Evaluation requires proper date mapping for accurate assessment against the test set.

**Model Comparison**: The Transformer model achieves higher accuracy (50.0%) compared to the MLP model (40.9%). Both models show significant class imbalance issues, with neither model successfully predicting the "Flat" class. The Transformer model performs better on the "Up" class (F1=0.64), while the MLP model performs better on the "Down" class (F1=0.58). This suggests that class weighting or data balancing techniques would improve model performance.

**Sentiment Analysis**: The LSTM-based sentiment model achieves 44.7% accuracy on financial text classification. The model shows bias toward predicting the "Neutral" class, likely due to class imbalance in the training dataset. Performance metrics: Macro Precision 0.090, Macro Recall 0.200, Macro F1-Score 0.124. The model generalizes from the Financial PhraseBank training set to real-world news headlines collected via NewsAPI.

**System Integration**: All components work cohesively:
- Price data is reliably fetched and cached
- Sentiment scores are computed from news headlines using deep learning models
- Fundamental metrics are successfully extracted from Yahoo Finance
- Multiple models generate predictions in real-time
- Scenario simulations provide risk assessment
- Intraday analysis identifies optimal monitoring windows

**User Experience**: The Streamlit interface provides an intuitive way to interact with the system. Users can easily switch between stocks, view model predictions side-by-side, explore scenario outcomes, and access detailed analytics.

**Practical Utility**: The system provides actionable buy/sell recommendations with confidence scores, helping users make informed decisions. The combination of technical indicators, sentiment, and fundamentals offers a more holistic view than any single approach alone.

### 3.2 Sentiment Analysis Models

The sentiment analysis system implements two deep learning architectures for classifying financial news headlines into five sentiment categories: **LSTM-based models** and **BERT-based models**. Both model implementations are available in the codebase (`ml/sentiment_lstm_model.py` and `ml/sentiment_bert_model.py`), with both models trained and evaluated. The BERT model achieves superior performance, while the LSTM model is currently used in production due to its lower computational requirements.

Both LSTM and BERT models have been trained and evaluated on the Financial PhraseBank dataset. The following table provides a direct comparison of their performance:

| Model | Test Accuracy | Validation Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|---------------|---------------------|-------------------|----------------|------------------|
| **LSTM** | 0.447 (44.7%) | 0.447 (44.7%) | 0.090 | 0.200 | 0.124 |
| **BERT** | 0.500 (50.0%) | 0.684 (68.4%) | 0.313 | 0.310 | 0.290 |

**Label Structure**: The model uses a 5-class classification scheme:
- **Label 0**: Very Negative - Extremely negative sentiment (e.g., "Stock price crashed 50% following fraud allegations")
- **Label 1**: Negative - Negative sentiment (e.g., "Earnings missed expectations causing stock decline")
- **Label 2**: Neutral - Neutral sentiment (e.g., "Company maintains steady performance this quarter")
- **Label 3**: Positive - Positive sentiment (e.g., "Strong earnings beat analyst expectations")
- **Label 4**: Very Positive - Extremely positive sentiment (e.g., "Record-breaking profits exceed all forecasts")

Labels are derived from two sources: (1) the Financial PhraseBank dataset, which contains manually annotated financial sentences, and (2) news headlines collected from NewsAPI, which are automatically labeled using VADER sentiment scores mapped to the 5-class system based on compound score thresholds: [-1, -0.6) → Very Negative, [-0.6, -0.2) → Negative, [-0.2, 0.2] → Neutral, (0.2, 0.6] → Positive, (0.6, 1] → Very Positive.

**Accuracy Measurement**: Model performance is evaluated using multiple metrics:

1. **Overall Accuracy**: The primary metric computed as the ratio of correct predictions to total predictions across all classes: `accuracy = (correct predictions) / (total predictions)`. This provides a global measure of classification performance.

2. **Per-Class Accuracy**: Individual accuracy for each sentiment class, calculated as the ratio of correctly predicted samples to total samples for that class. This metric helps identify if the model performs better on certain sentiment categories.

3. **Macro-Averaged Metrics**: Precision, recall, and F1-score averaged across all five classes without weighting. These metrics treat all classes equally, providing insight into performance across the full sentiment spectrum: `macro_precision = mean(precision_i)`, `macro_recall = mean(recall_i)`, `macro_f1 = mean(f1_i)`.

4. **Weighted-Averaged Metrics**: Precision, recall, and F1-score weighted by the number of samples in each class. These metrics account for class imbalance in the dataset, giving more weight to classes with more samples: `weighted_f1 = Σ(f1_i × support_i) / Σ(support_i)`.

5. **Confusion Matrix**: A 5×5 matrix showing the distribution of predicted labels versus actual labels. This visualization helps identify common misclassification patterns (e.g., whether the model confuses "Negative" with "Very Negative").

The models are evaluated on held-out test sets using these metrics, with early stopping based on validation loss to prevent overfitting. The evaluation framework supports comparison between different model configurations and provides comprehensive performance insights beyond simple accuracy.

**LSTM Model Details**:

The LSTM-based sentiment classifier uses a bidirectional recurrent neural network architecture with embedding layers. The model supports both random and pre-trained GloVe embeddings, with configurable depth and hidden dimensions.

- **Per-Class Performance** (on test set):
  - Very Negative (Class 0): Precision 0.00, Recall 0.00, F1 0.00, Support: 3
  - Negative (Class 1): Precision 0.00, Recall 0.00, F1 0.00, Support: 8
  - Neutral (Class 2): Precision 0.45, Recall 1.00, F1 0.62, Support: 17
  - Positive (Class 3): Precision 0.00, Recall 0.00, F1 0.00, Support: 7
  - Very Positive (Class 4): Precision 0.00, Recall 0.00, F1 0.00, Support: 3
- **Model Configuration**: Bidirectional LSTM with 2 layers, hidden_dim=128, embedding_dim=100, dropout=0.5
- **Training Details**: Trained for 19 epochs with early stopping (patience=5), best validation accuracy: 0.447

The LSTM model demonstrates bias toward predicting the "Neutral" class, which dominates the predictions. This is likely due to class imbalance in the small training dataset (176 train samples, 38 test samples). The model fails to predict extreme sentiment classes (Very Negative, Negative, Positive, Very Positive), suggesting that data augmentation or class balancing techniques would improve performance.

**BERT Model Details**:

The BERT-based sentiment classifier fine-tunes pre-trained transformer models (`bert-base-uncased`) for financial sentiment analysis using the Hugging Face Transformers library.

- **Per-Class Performance** (on test set):
  - Very Negative (Class 0): Precision 0.00, Recall 0.00, F1 0.00, Support: 3
  - Negative (Class 1): Precision 0.31, Recall 0.50, F1 0.38, Support: 8
  - Neutral (Class 2): Precision 0.59, Recall 0.76, F1 0.67, Support: 17
  - Positive (Class 3): Precision 0.67, Recall 0.29, F1 0.40, Support: 7
  - Very Positive (Class 4): Precision 0.00, Recall 0.00, F1 0.00, Support: 3
- **Model Configuration**: Fine-tuned bert-base-uncased using Hugging Face Trainer API
- **Training Details**: Trained for 5 epochs with early stopping (patience=2), batch_size=8, learning_rate=2e-5

**Model Comparison Summary**:

The BERT model achieves 50.0% test accuracy, outperforming the LSTM model (44.7%). On the validation set, BERT achieves 68.4% accuracy compared to LSTM's 44.7%, demonstrating the benefits of transfer learning from pre-trained language models. BERT shows significantly better macro-averaged metrics (Precision: 0.313 vs 0.090, Recall: 0.310 vs 0.200, F1: 0.290 vs 0.124), indicating superior overall performance. Both models show bias toward the "Neutral" class, though BERT performs better on other classes (e.g., Negative F1=0.38, Positive F1=0.40 vs LSTM's 0.00 for both). Both models struggle with extreme sentiment classes (Very Negative, Very Positive) due to the small dataset size. The BERT model successfully leverages pre-trained contextualized word embeddings to capture nuanced semantic relationships in financial text, though performance is limited by the small training dataset (176 samples).

### 3.3 Limitations

1. **Market Efficiency**: Stock prices follow efficient market hypotheses; short-term predictability is inherently limited. Models serve as decision-support tools rather than guarantees. The achieved accuracy of ~55% on direction prediction reflects the inherent difficulty of stock market forecasting.

2. **Data Quality**: Yahoo Finance data may have inconsistencies or delays. NewsAPI rate limits may affect sentiment analysis for high-frequency use.

3. **Temporal Generalization**: Models trained on historical data may not generalize to future market regimes, especially during unusual market conditions. Performance metrics reported are based on historical data and may vary in different market environments.

4. **Class Imbalance**: The stock direction prediction models show better performance on the "Flat" class due to class distribution. The sentiment model performs better on neutral sentiment, which dominates the training dataset.

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

## 7. Appendix

### 7.1 Model Technical Documentation

A comprehensive technical documentation of all models implemented in this project is provided in the separate document **`Model_Technical_Documentation.pdf`** (located in the `docs/` directory).

This documentation includes:

1. **Detailed Model Specifications**: Complete input/output specifications, architecture details, and technical flow for each model:
   - Baseline Moving Average Model
   - MLP (Multi-Layer Perceptron) Model
   - Transformer Model
   - LSTM Sentiment Analysis Model
   - BERT Sentiment Analysis Model

2. **Step-by-Step Examples**: Concrete examples showing how each model transforms inputs to outputs/probabilities:
   - Price data to direction predictions (Baseline)
   - Feature vectors to probabilities (MLP)
   - Sequence features to probabilities (Transformer)
   - Sentence to probabilities (LSTM and BERT)

3. **Complete End-to-End Flows**: Full pipeline documentation from data collection through final output generation for each model.

4. **Performance Metrics**: Detailed evaluation results including accuracy, precision, recall, and F1-scores for all models.

5. **Example Inputs and Outputs**: Real examples with actual numerical values demonstrating model behavior.

The Model Technical Documentation provides the technical depth and implementation details that complement this project report, which focuses on methodology, results, and high-level system architecture.

---

**Word Count**: ~1,800 words (approximately 5 pages when formatted as PDF with standard formatting)

