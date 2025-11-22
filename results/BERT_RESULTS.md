# BERT Sentiment Model Results

## Training Summary

**Model**: bert-base-uncased  
**Training Date**: November 19, 2025  
**Dataset**: Financial PhraseBank + NewsAPI headlines

## Performance Metrics

### Validation Set
- **Accuracy**: 68.4% (0.684)
- **F1-Score (Macro)**: 0.379
- **Best Epoch**: 4

### Test Set
- **Accuracy**: 50.0% (0.500)
- **Note**: Test set is small (38 samples), results may vary

## Training Configuration

- **Base Model**: bert-base-uncased
- **Epochs**: 5 (with early stopping)
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens
- **Patience**: 2 epochs

## Comparison with LSTM

| Model | Validation Accuracy | Test Accuracy | Notes |
|-------|-------------------|---------------|-------|
| **BERT** | 68.4% | 50.0% | Pre-trained, better validation performance |
| **LSTM** | 44.7% | 44.7% | Trained from scratch, consistent performance |

## Key Findings

1. **BERT outperforms LSTM on validation set** (68.4% vs 44.7%)
2. **Test set performance** is similar (50.0% vs 44.7%), but test set is very small
3. **Transfer learning benefits**: BERT leverages pre-trained knowledge
4. **Small dataset limitation**: Both models affected by limited training data (176 samples)

## Model Location

Trained model saved to: `models/bert_sentiment/final_model/`

## Usage

```python
from ml.sentiment_bert_model import load_bert_model, predict_with_bert

model, tokenizer = load_bert_model("models/bert_sentiment/final_model")
predictions = predict_with_bert(model, tokenizer, ["Your text here"])
```




