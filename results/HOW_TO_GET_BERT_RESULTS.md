# How to Get BERT Sentiment Model Results

## Current Status
- ✅ BERT code is fully implemented (`ml/sentiment_bert_model.py`)
- ✅ Training script exists (`ml/train_bert_sentiment.py`)
- ❌ Model has not been trained yet
- ❌ Missing dependency: `accelerate` library

## Steps to Train BERT and Get Results

### 1. Install Required Dependencies

```bash
pip install accelerate>=0.26.0
# or
pip install transformers[torch]
```

### 2. Train the BERT Model

```bash
cd /Users/waiwai/Desktop/AI_Stocks
python3 -m ml.train_bert_sentiment \
    --model-name bert-base-uncased \
    --epochs 10 \
    --patience 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --output models/bert_sentiment
```

### 3. Expected Output

After training, you should get:
- Model saved to `models/bert_sentiment/final_model/`
- Validation accuracy
- Test accuracy
- Performance metrics (precision, recall, F1-score)

### 4. Evaluate the Trained Model

Use the evaluation script:

```bash
python3 results/get_all_model_results.py
```

This will evaluate BERT along with other models and show:
- Test accuracy
- Macro-averaged precision, recall, F1-score
- Per-class performance metrics

## Alternative: Use Pre-trained Financial BERT

For better financial domain performance, you can use a financial BERT:

```bash
python3 -m ml.train_bert_sentiment \
    --model-name yiyanghkust/finbert-pretrain \
    --epochs 10 \
    --patience 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --output models/bert_sentiment_financial
```

## Expected Performance

Based on the implementation and typical BERT performance:
- **Expected Accuracy**: 50-70% (depending on dataset and training)
- **Advantages over LSTM**: 
  - Pre-trained language understanding
  - Better context awareness
  - Transfer learning benefits

## Current Dataset Size

- Train: 176 samples
- Validation: 38 samples  
- Test: 38 samples

**Note**: Small dataset size may limit BERT's potential. Consider:
- Data augmentation
- Using more training data
- Transfer learning from larger financial text datasets




