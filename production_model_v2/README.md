
# Meme Sentiment Analysis Model

## Model Description
This model is a fine-tuned DistilBERT model for sentiment analysis of meme text.

## Training Data
- **Total samples**: 12
- **Classes**: ['negative', 'neutral', 'positive']
- **Text preprocessing**: Advanced OCR with multiple preprocessing techniques

## Model Performance
- **Accuracy**: 0.417
- **F1-Score**: 0.245

## Usage
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('production_model_v2')
model = DistilBertForSequenceClassification.from_pretrained('production_model_v2')
```

## Training Details
- **Model**: distilbert-base-uncased
- **Training date**: 2025-07-05
- **Framework**: Transformers + PyTorch

## Limitations
This model is specifically trained for meme text sentiment analysis and may not generalize well to other domains.
        