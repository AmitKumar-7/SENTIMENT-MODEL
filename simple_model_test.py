#!/usr/bin/env python3
"""
Simple test script for the saved sentiment model
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime

def test_saved_model():
    """Test the saved model with direct inference"""
    print("🎯 Testing Saved Sentiment Model")
    print("="*50)
    
    model_path = "./production_model_v2"
    
    try:
        # Load model and tokenizer
        print("🔄 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Model loaded successfully!")
        
        # Model info
        print(f"\n📊 Model Info:")
        print(f"   Model type: {model.config.model_type}")
        print(f"   Number of labels: {model.config.num_labels}")
        print(f"   Vocabulary size: {len(tokenizer.vocab)}")
        
        # Test texts
        test_texts = [
            "This meme is absolutely hilarious! 😂",
            "I hate this stupid meme",
            "This is just a regular meme", 
            "Best meme ever! Love it!",
            "This meme is okay, nothing special",
            "Worst meme I've ever seen",
            "LOL this is so funny!",
            "This makes me sad",
            "Meh, whatever"
        ]
        
        print("\n🧪 Testing predictions:")
        print("-" * 50)
        
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        
        # Make predictions
        for i, text in enumerate(test_texts, 1):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            sentiment = sentiment_labels[predicted_class]
            
            # Show all probabilities
            all_probs = [f"{sentiment_labels[j]}: {probabilities[0][j].item():.3f}" for j in range(3)]
            
            print(f"{i:2d}. '{text[:40]}{'...' if len(text) > 40 else ''}'")
            print(f"    → {sentiment} (confidence: {confidence:.3f})")
            print(f"    → All scores: {', '.join(all_probs)}")
            print()
        
        # Test batch processing
        print("\n📦 Testing batch processing:")
        print("-" * 50)
        
        batch_texts = test_texts[:3]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
        
        for i, text in enumerate(batch_texts):
            predicted_class = predicted_classes[i].item()
            confidence = probabilities[i][predicted_class].item()
            sentiment = sentiment_labels[predicted_class]
            
            print(f"{i+1}. '{text[:30]}...' → {sentiment} ({confidence:.3f})")
        
        print("\n✅ Model testing completed successfully!")
        print(f"✅ Model is saved and working correctly at: {model_path}")
        
        # Save a simple usage example
        with open("model_usage_example.py", "w") as f:
            f.write('''# Simple usage example for the saved model
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
''')
        
        print("📄 Usage example saved to: model_usage_example.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_saved_model()
