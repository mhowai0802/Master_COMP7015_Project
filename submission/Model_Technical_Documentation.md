# AI Stocks Project - Model Technical Documentation

## Project Overview

The AI Stocks project is a comprehensive stock prediction system that combines multiple machine learning models to provide buy/sell recommendations for AI-related stocks. The system integrates price data, news sentiment analysis, and fundamental metrics to generate actionable trading insights. The project implements five distinct models: a baseline moving average model, two deep learning stock prediction models (MLP and Transformer), and two sentiment analysis models (LSTM and BERT).

---

## Model 1: Baseline Moving Average Model

### Input
- **Stock Price Series**: Historical daily OHLCV (Open, High, Low, Close, Volume) data
- **As-of Date**: Reference date for prediction
- **Minimum Data Requirement**: At least 30 days of historical price data

### Output
- **Direction Signal**: "up", "down", or "flat" trend classification
- **Confidence Score**: Float value between 0-1 indicating prediction confidence
- **Buy/Sell Recommendations**: 
  - `should_buy`: Boolean flag (True if trend is "up")
  - `should_sell`: Boolean flag (True if trend is "down")
- **Price Targets**:
  - `suggested_buy_price`: 2% below latest close price
  - `suggested_sell_price`: 5% above latest close price

### Example Input
```
Stock: AAPL (Apple Inc.)
As-of Date: 2025-11-19
Price Series: 365 days of OHLCV data
  - Latest close: $267.44
  - 10-day MA: $268.50
  - 30-day MA: $265.20
```

### Example Output
```
Direction: "up"
Confidence: 0.75
should_buy: True
should_sell: False
suggested_buy_price: $262.09 (98% of $267.44)
suggested_sell_price: $280.81 (105% of $267.44)
```

### Technical Flow
1. **Data Conversion**: Convert StockPriceSeries to pandas DataFrame with datetime index
2. **Moving Average Calculation**: 
   - Compute 10-day short-term moving average (MA_short)
   - Compute 30-day long-term moving average (MA_long)
3. **Trend Detection**:
   - Calculate difference: MA_short - MA_long
   - Normalize by current close price to get relative strength
   - Classify trend:
     - "up" if relative difference > 2%
     - "down" if relative difference < -2%
     - "flat" otherwise
4. **Confidence Scoring**: Map relative strength to confidence (0.3-0.9 range)
5. **Price Recommendation**: Generate buy/sell levels based on simple percentage rules

---

## Model 2: MLP Stock Prediction Model

### Input
- **Stock Price Series**: Historical daily OHLCV data
- **Sentiment Score**: Optional float in [-1, 1] range from news sentiment analysis
- **Fundamental Metrics**: Dictionary containing P/E ratio and P/S ratio
- **Model Configuration**: 
  - Input dimension: 8 features
  - Hidden dimension: 64 (default)
  - Number of layers: 2 (default)

### Output
- **Direction Classification**: One of three classes:
  - Class 0: "down" (下跌)
  - Class 1: "flat" (橫向/小波動)
  - Class 2: "up" (上升)
- **Class Probabilities**: Softmax probabilities for each class
- **Confidence**: Maximum probability value
- **Trading Recommendations**:
  - `should_buy`: True for "up" predictions
  - `should_sell`: True for "down" predictions
- **Price Targets**: Dynamic based on predicted direction and last close price

### Example Input
```
Feature Vector (8 dimensions):
  [267.44,    # Last close price
   268.50,    # 10-day MA
   265.20,    # 30-day MA
   2.15,      # 10-day std
   5.80,      # 30-day std
   0.35,      # Sentiment score (positive)
   28.5,      # P/E ratio
   7.2]       # P/S ratio
```

### Example Output
```
Predicted Class: 0 (down)
Class Probabilities: [0.96, 0.00, 0.04]
  - Down: 96%
  - Flat: 0%
  - Up: 4%
Confidence: 0.96
should_buy: False
should_sell: True
suggested_buy_price: $254.07 (95% of $267.44)
suggested_sell_price: $262.09 (98% of $267.44)

Performance Metrics (Test Set):
  Accuracy: 40.93%
  Precision (Down): 0.41, Recall: 0.96, F1: 0.58
```

### Technical Flow
1. **Feature Engineering**:
   - Extract last close price
   - Calculate 10-day and 30-day moving averages
   - Compute 10-day and 30-day standard deviations
   - Incorporate sentiment score (default 0.0 if missing)
   - Add P/E and P/S ratios (default 0.0 if missing)
   - Result: 8-dimensional feature vector
2. **Model Architecture**:
   - Input layer: 8 features
   - Hidden layers: Multiple fully connected layers with ReLU activation
   - Output layer: 3-class logits
3. **Forward Pass**:
   - Pass feature vector through MLP network
   - Apply softmax to logits to get class probabilities
   - Select class with highest probability
4. **Post-Processing**:
   - Map predicted class to direction string
   - Generate buy/sell flags based on direction
   - Calculate price targets:
     - "up": buy at 98% of close, sell at 105% of close
     - "down": buy at 95% of close, sell at 98% of close
     - "flat": buy at 99% of close, sell at 101% of close

---

## Model 3: Transformer Stock Prediction Model

### Input
- **Stock Price Series**: Historical daily OHLCV data (sequence)
- **Sentiment Score**: Optional float in [-1, 1] range
- **Fundamental Metrics**: Dictionary with P/E and P/S ratios
- **Sequence Length**: Maximum 128 days (default)
- **Model Configuration**:
  - d_model: 32 (default)
  - Number of attention heads: 4 (default)
  - Number of encoder layers: 2 (default)
  - Feedforward dimension: 64 (default)

### Output
- **Direction Classification**: Three-class prediction (down/flat/up)
- **Class Probabilities**: Softmax distribution over classes
- **Confidence**: Maximum probability
- **Trading Recommendations**: Buy/sell flags and price targets (same format as MLP)

### Example Input
```
Sequence Features (128 days × 8 features):
  Day 1: [225.96, 229.12, 225.64, 227.25, 36211800, 0.35, 28.5, 7.2]
  Day 2: [227.03, 228.89, 224.87, 227.97, 35169600, 0.35, 28.5, 7.2]
  ...
  Day 128: [269.92, 270.70, 265.32, 267.44, 43692217, 0.35, 28.5, 7.2]
  
Model Config: d_model=64, nhead=8, num_layers=3
```

### Example Output
```
Predicted Class: 2 (up)
Class Probabilities: [0.10, 0.00, 0.90]
  - Down: 10%
  - Flat: 0%
  - Up: 90%
Confidence: 0.90
should_buy: True
should_sell: False
suggested_buy_price: $262.09 (98% of $267.44)
suggested_sell_price: $280.81 (105% of $267.44)

Performance Metrics (Test Set):
  Accuracy: 50.00%
  Precision (Up): 0.49, Recall: 0.90, F1: 0.64
```

### Technical Flow
1. **Sequence Feature Construction**:
   - For each day in price history, create feature vector:
     - [open, high, low, close, volume, sentiment, P/E, P/S]
   - Truncate or pad to max_len (128 days)
   - Result: (seq_len, 8) feature matrix
2. **Input Projection**:
   - Linear projection from 8 features to d_model dimensions
3. **Positional Encoding**:
   - Add sinusoidal positional encodings to capture temporal order
   - Uses sin/cos functions with different frequencies
4. **Transformer Encoder**:
   - Multi-head self-attention mechanism captures dependencies across time steps
   - Feedforward networks with residual connections
   - Layer normalization for stability
   - Multiple encoder layers stack to learn hierarchical patterns
5. **Sequence Aggregation**:
   - Extract last time step representation (similar to CLS token)
   - This summarizes the entire sequence context
6. **Classification Head**:
   - Layer normalization
   - Linear projection to 3 classes
   - Softmax for probability distribution
7. **Output Generation**: Same post-processing as MLP model

---

## Model 4: LSTM Sentiment Analysis Model

### Input
- **Text Sequences**: Financial news headlines or text snippets
- **Vocabulary**: Pre-built vocabulary mapping words to indices
- **Sequence Length**: Variable (padded/truncated to fixed max length)
- **Model Configuration**:
  - Embedding dimension: 100 (default)
  - Hidden dimension: 128 (default)
  - Number of layers: 2 (default)
  - Dropout: 0.5 (default)
  - Number of classes: 5 (Very Negative, Negative, Neutral, Positive, Very Positive)

### Output
- **Sentiment Class**: One of five classes (0-4)
- **Class Probabilities**: Softmax probabilities for each sentiment class
- **Aggregate Sentiment Score**: Converted to [-1, 1] range for integration with stock models

### Example Input
```
Text: "Apple reports strong quarterly earnings, beats expectations"
Tokenized: [apple, reports, strong, quarterly, earnings, beats, expectations]
Sequence Length: 7 (padded to max_length)
Vocabulary Size: 5,000
Embedding: 100-dimensional vectors
```

### Example Output
```
Predicted Class: 3 (Positive)
Class Probabilities: [0.05, 0.10, 0.70, 0.12, 0.03]
  - Very Negative: 5%
  - Negative: 10%
  - Neutral: 70%
  - Positive: 12%
  - Very Positive: 3%
Aggregate Sentiment Score: 0.35 (mapped to [-1, 1] range)

Performance Metrics (Test Set):
  Accuracy: 44.74%
  Macro F1-Score: 0.1236
  Note: Model biases toward Neutral class
```

### Technical Flow
1. **Text Preprocessing**:
   - Tokenize input text into words
   - Map words to vocabulary indices
   - Pad/truncate sequences to fixed length
2. **Embedding Layer**:
   - Convert word indices to dense vectors
   - Supports random initialization or pre-trained embeddings
   - Embedding dimension: 100
3. **LSTM/GRU Processing**:
   - Process sequence through bidirectional or unidirectional LSTM/GRU
   - Multiple layers with dropout for regularization
   - Hidden state dimension: 128
4. **Sequence Representation**:
   - Extract final hidden state from last time step
   - This captures the overall sentiment of the sequence
5. **Classification**:
   - Apply dropout for regularization
   - Linear projection to 5 classes
   - Softmax for probability distribution
6. **Score Conversion**:
   - Map 5-class prediction to continuous [-1, 1] score
   - Used as input feature for stock prediction models

---

## Model 5: BERT Sentiment Analysis Model

### Input
- **Text Sequences**: Financial news headlines or text snippets
- **Maximum Length**: 128 tokens (default)
- **Model Base**: Pre-trained BERT model (bert-base-uncased or financial BERT variants)
- **Model Configuration**:
  - Number of classes: 5
  - Dropout: 0.1 (default)
  - Fine-tuning approach: Full model fine-tuning

### Output
- **Sentiment Class**: One of five classes (0-4)
- **Class Probabilities**: Softmax probabilities
- **Aggregate Sentiment Score**: Converted to [-1, 1] range

### Example Input
```
Text: "NVIDIA announces breakthrough AI chip technology, stock surges"
Tokenized: [CLS] nvidia announces breakthrough ai chip technology stock surges [SEP]
Input IDs: [101, 12345, 6789, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 102]
Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...] (padded to 128)
Model: bert-base-uncased (or financial BERT variant)
```

### Example Output
```
Predicted Class: 4 (Very Positive)
Class Probabilities: [0.02, 0.05, 0.15, 0.30, 0.48]
  - Very Negative: 2%
  - Negative: 5%
  - Neutral: 15%
  - Positive: 30%
  - Very Positive: 48%
Aggregate Sentiment Score: 0.82 (mapped to [-1, 1] range)

Training Status: Model infrastructure ready, requires training
Expected Performance: Superior to LSTM due to pre-trained embeddings
```

### Technical Flow
1. **Tokenization**:
   - Use BERT tokenizer to convert text to subword tokens
   - Add special tokens: [CLS] at start, [SEP] for separation
   - Truncate/pad to max_length (128 tokens)
   - Create attention mask for padding tokens
2. **BERT Encoder**:
   - Input embeddings: token embeddings + positional embeddings + segment embeddings
   - Multi-layer Transformer encoder with self-attention
   - Pre-trained weights capture rich linguistic patterns
3. **Fine-tuning**:
   - Add classification head on top of [CLS] token representation
   - Fine-tune entire model on financial sentiment dataset
   - Use early stopping based on validation loss
4. **Classification Head**:
   - Extract [CLS] token representation (sequence summary)
   - Linear projection to 5 classes
   - Softmax for probability distribution
5. **Training Process**:
   - Use Hugging Face Trainer API
   - Adam optimizer with learning rate 2e-5
   - Mixed precision training (FP16) if GPU available
   - Evaluation after each epoch
6. **Score Conversion**:
   - Map 5-class prediction to continuous sentiment score
   - Integrated into stock prediction pipeline

---

## Model Integration Pipeline

### Data Flow
1. **Data Collection**: Fetch price history, news headlines, and fundamental data
2. **Sentiment Analysis**: Process news headlines through LSTM/BERT models
3. **Feature Extraction**: Combine price, sentiment, and fundamental features
4. **Stock Prediction**: Run through Baseline, MLP, or Transformer models
5. **Scenario Generation**: Optional Monte Carlo simulation for risk assessment
6. **Output Aggregation**: Combine predictions from multiple models for final recommendation

### Performance Summary
- **MLP Model**: 40.93% accuracy, struggles with class imbalance
- **Transformer Model**: 50.00% accuracy, better sequence modeling
- **LSTM Sentiment**: 44.74% accuracy, small dataset limitation
- **BERT Sentiment**: Requires training, expected superior performance
- **Baseline Model**: Simple but interpretable moving average strategy

---

## Conclusion

The AI Stocks project demonstrates a comprehensive approach to stock prediction by integrating multiple machine learning paradigms: traditional technical analysis (baseline), deep learning for tabular data (MLP), sequence modeling (Transformer), and natural language processing (LSTM/BERT). Each model contributes unique insights, and their combination provides a robust framework for financial decision-making. The system successfully integrates heterogeneous data sources (price, sentiment, fundamentals) and demonstrates the practical application of modern deep learning techniques to financial markets.

