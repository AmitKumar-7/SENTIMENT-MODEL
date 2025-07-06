# Simple usage example for the saved model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model
model_path = "./production_model_v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    """Predict sentiment of text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    return sentiment_labels[predicted_class], confidence

# Example usage
text = "This meme is hilarious!"
sentiment, confidence = predict_sentiment(text)
print(f"Text: {text}")
print(f"Sentiment: {sentiment} (confidence: {confidence:.3f})")
