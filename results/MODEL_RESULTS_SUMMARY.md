# Comprehensive Model Results Summary

## Stock Prediction Models

### 1. MLP Model
- **Test Accuracy**: 40.93%
- **Macro Precision**: 0.2483
- **Macro Recall**: 0.3250
- **Macro F1-Score**: 0.2039
- **Best Configuration**: hidden_dim=64, num_layers=3
- **Per-Class Performance**:
  - Down (Class 0): Precision 0.41, Recall 0.96, F1 0.58
  - Flat (Class 1): Precision 0.00, Recall 0.00, F1 0.00 (not predicted)
  - Up (Class 2): Precision 0.33, Recall 0.02, F1 0.04
- **Issue**: Model heavily biases toward predicting "Down" class, fails to predict "Flat" class

### 2. Transformer Model
- **Test Accuracy**: 50.00%
- **Macro Precision**: 0.3475
- **Macro Recall**: 0.3530
- **Macro F1-Score**: 0.2935
- **Best Configuration**: d_model=64, nhead=8, num_layers=3, dim_feedforward=128
- **Per-Class Performance**:
  - Down (Class 0): Precision 0.55, Recall 0.16, F1 0.24
  - Flat (Class 1): Precision 0.00, Recall 0.00, F1 0.00 (not predicted)
  - Up (Class 2): Precision 0.49, Recall 0.90, F1 0.64
- **Issue**: Model heavily biases toward predicting "Up" class, fails to predict "Flat" class

### 3. Baseline Model
- **Test Accuracy**: Evaluation needs improvement
- **Model Type**: Moving average crossover strategy
- **Note**: Baseline evaluation requires proper date mapping for accurate assessment

## Sentiment Analysis Models

### 4. LSTM Sentiment Model
- **Test Accuracy**: 44.74%
- **Macro Precision**: 0.0895
- **Macro Recall**: 0.2000
- **Macro F1-Score**: 0.1236
- **Model Configuration**: Bidirectional LSTM, 2 layers, hidden_dim=128, embedding_dim=100, dropout=0.5
- **Issue**: Model heavily biases toward predicting "Neutral" class

### 5. BERT Sentiment Model
- **Status**: Not trained yet
- **Implementation**: Fully implemented in `ml/sentiment_bert_model.py`
- **Note**: BERT model code exists but requires training before evaluation

## Dataset Information

### Stock Prediction Dataset
- **Total Samples**: 2,150
- **Train Samples**: 1,720 (80%)
- **Test Samples**: 430 (20%)
- **Stocks**: 10 AI-related stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, TSM, SMCI)
- **Horizon**: 5-day future return prediction
- **Classes**: Down, Flat, Up (threshold: ±1%)

### Sentiment Analysis Dataset
- **Train Samples**: 176
- **Val Samples**: 38
- **Test Samples**: 38
- **Total**: 252 samples
- **Classes**: 5-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive)
- **Source**: Financial PhraseBank + NewsAPI headlines

## Key Observations

1. **Class Imbalance**: Both stock prediction models struggle with the "Flat" class, likely due to class imbalance
2. **Transformer Outperforms MLP**: Transformer achieves 50% accuracy vs MLP's 40.93%
3. **Sentiment Models**: LSTM sentiment model shows poor performance, likely due to small dataset size
4. **BERT Not Trained**: BERT sentiment model infrastructure exists but hasn't been trained

## Recommendations

1. Address class imbalance in stock prediction models (e.g., class weighting, oversampling)
2. Train BERT sentiment model for comparison with LSTM
3. Improve baseline model evaluation methodology
4. Consider data augmentation for sentiment analysis dataset

