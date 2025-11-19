# Model Evaluation Results

This folder contains the evaluation results and statistics from running the complete project.

## Files

- **get_detailed_stats.py**: Python script used to generate detailed evaluation statistics
- **evaluation_results.log**: Comprehensive evaluation results for MLP, Transformer, and Baseline models
- **detailed_stats.log**: Detailed per-class metrics, confusion matrices, and performance statistics for MLP and Transformer models
- **sentiment_training.log**: Training output for the LSTM sentiment analysis model

## Summary of Results

### Stock Prediction Models

**MLP Model:**
- Test Accuracy: 40.93%
- Macro Precision: 0.2483
- Macro Recall: 0.3250
- Macro F1-Score: 0.2039
- Best Configuration: hidden_dim=64, num_layers=3

**Transformer Model:**
- Test Accuracy: 50.00%
- Macro Precision: 0.3475
- Macro Recall: 0.3530
- Macro F1-Score: 0.2935
- Best Configuration: d_model=64, nhead=8, num_layers=3, dim_feedforward=128

### Sentiment Analysis Model

**LSTM Model:**
- Test Accuracy: 44.74%
- Best Validation Accuracy: 44.74%

## Notes

- Models were trained on 10 AI-related stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, TSM, SMCI)
- Dataset size: 2150 samples (1720 train, 430 test)
- Evaluation date: November 19, 2025

