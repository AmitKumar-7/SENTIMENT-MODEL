#!/usr/bin/env python3
"""
Test Improved Model - Compare with old model performance
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from datetime import datetime

class ImprovedModelTester:
    """Test the improved model and compare with old model"""
    
    def __init__(self, improved_model_path='./improved_model_final'):
        self.improved_model_path = improved_model_path
        self.label_to_id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.id_to_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        # Load improved model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(improved_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(improved_model_path)
            self.model.eval()
            print(f"✅ Loaded improved model from: {improved_model_path}")
        except Exception as e:
            print(f"❌ Error loading improved model: {e}")
            self.model = None
    
    def predict_sentiment(self, text):
        """Predict sentiment with improved model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            sentiment = self.id_to_label[predicted_class]
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'all_scores': {
                    'Negative': predictions[0][0].item(),
                    'Neutral': predictions[0][1].item(),
                    'Positive': predictions[0][2].item()
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_improved_model(self):
        """Test improved model performance"""
        print("\n🧪 Testing Improved Model")
        print("=" * 50)
        
        test_cases = [
            {"text": "This meme is absolutely hilarious! 😂😂😂", "expected": "Positive"},
            {"text": "I hate everything about this stupid day", "expected": "Negative"},
            {"text": "It's Wednesday my dudes", "expected": "Neutral"},
            {"text": "Best meme ever! Love it so much!", "expected": "Positive"},
            {"text": "Ugh, another boring meeting", "expected": "Negative"},
            {"text": "Today is Thursday", "expected": "Neutral"},
            {"text": "This is amazing! 🔥💯", "expected": "Positive"},
            {"text": "This sucks so much 👎", "expected": "Negative"},
            {"text": "Just another regular day", "expected": "Neutral"},
            {"text": "Love this wholesome content ❤️", "expected": "Positive"},
            {"text": "Terrible meme! Not funny at all", "expected": "Negative"},
            {"text": "Some random meme content", "expected": "Neutral"},
            {"text": "Absolutely brilliant! Outstanding work", "expected": "Positive"},
            {"text": "This is awful and disappointing", "expected": "Negative"},
            {"text": "Here is some ordinary content", "expected": "Neutral"}
        ]
        
        correct = 0
        total = len(test_cases)
        results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {case['text']}")
            result = self.predict_sentiment(case['text'])
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
                
            predicted = result['sentiment']
            expected = case['expected']
            confidence = result['confidence']
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
                print(f"✅ Predicted: {predicted} (Expected: {expected}) - Confidence: {confidence:.2%}")
            else:
                print(f"❌ Predicted: {predicted} (Expected: {expected}) - Confidence: {confidence:.2%}")
            
            results.append({
                'text': case['text'],
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct,
                'all_scores': result['all_scores']
            })
        
        accuracy = correct / total
        print(f"\n📊 Improved Model Results:")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return results, accuracy
    
    def compare_with_old_model(self):
        """Compare improved model with old model results"""
        print("\n📊 Comparison: Improved vs Old Model")
        print("=" * 60)
        
        # Test improved model
        improved_results, improved_accuracy = self.test_improved_model()
        
        # Old model results (from previous testing)
        old_accuracy = 0.30  # 30% from previous tests
        
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        print("=" * 40)
        print(f"Old Model Accuracy:      {old_accuracy:.2%} ❌")
        print(f"Improved Model Accuracy: {improved_accuracy:.2%} ✅")
        print(f"Improvement:             {(improved_accuracy - old_accuracy):.2%}")
        print(f"Relative Improvement:    {(improved_accuracy / old_accuracy):.1f}x better")
        
        # Detailed comparison
        print(f"\n📋 Detailed Analysis:")
        print("=" * 40)
        
        if improved_accuracy > 0.75:
            print("✅ EXCELLENT: Model accuracy > 75% - Ready for production!")
        elif improved_accuracy > 0.60:
            print("✅ GOOD: Model accuracy > 60% - Suitable for most use cases")
        elif improved_accuracy > 0.45:
            print("⚠️ FAIR: Model accuracy > 45% - Better than old model but needs improvement")
        else:
            print("❌ POOR: Model accuracy < 45% - Still needs work")
        
        # Confidence analysis
        if improved_results:
            confidences = [r['confidence'] for r in improved_results if r['correct']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"\n🎯 Confidence Analysis:")
            print(f"Average confidence for correct predictions: {avg_confidence:.2%}")
            
            if avg_confidence > 0.80:
                print("✅ HIGH CONFIDENCE: Model is very certain about correct predictions")
            elif avg_confidence > 0.60:
                print("✅ MODERATE CONFIDENCE: Model is reasonably certain")
            else:
                print("⚠️ LOW CONFIDENCE: Model is uncertain about predictions")
        
        return improved_accuracy

def main():
    """Main testing function"""
    print("🔬 IMPROVED MODEL TESTING")
    print("=" * 50)
    print("Testing the improved model and comparing with old performance")
    print()
    
    # Test improved model
    tester = ImprovedModelTester()
    
    if tester.model is None:
        print("❌ Cannot test - improved model not found!")
        print("📋 Please run 'python fix_model_training.py' first to train the improved model")
        return
    
    # Run comparison
    final_accuracy = tester.compare_with_old_model()
    
    print(f"\n🎯 TESTING COMPLETED!")
    print("=" * 50)
    
    if final_accuracy > 0.75:
        print("🎉 SUCCESS: Your improved model is working great!")
        print("✅ Ready for production use")
        print("📋 Next steps:")
        print("1. Deploy the improved model")
        print("2. Update your analyzer to use './improved_model_final'")
        print("3. Monitor performance in production")
    elif final_accuracy > 0.45:
        print("✅ IMPROVEMENT: Much better than the old model!")
        print("⚠️ Consider collecting more training data for even better results")
        print("📋 Next steps:")
        print("1. Use this improved model instead of the old one")
        print("2. Collect more balanced training data")
        print("3. Retrain with larger dataset")
    else:
        print("⚠️ PARTIAL SUCCESS: Some improvement but still needs work")
        print("📋 Recommendations:")
        print("1. Collect more high-quality training data")
        print("2. Try different model architectures")
        print("3. Use ensemble methods")
    
    return final_accuracy

if __name__ == "__main__":
    main()
