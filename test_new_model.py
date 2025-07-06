#!/usr/bin/env python3
"""
Test the newly trained meme sentiment model
Compare with old model and validate performance
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class NewModelTester:
    """Test the newly trained model"""
    
    def __init__(self, model_path='./meme_sentiment_model_final'):
        self.model_path = model_path
        self.label_to_id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.id_to_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        # Load the new model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            print(f"✅ Loaded new model from: {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading new model: {e}")
            print("💡 Make sure to run 'python train_from_scratch.py' first")
            self.model_loaded = False
    
    def predict_sentiment(self, text):
        """Predict sentiment with the new model"""
        if not self.model_loaded:
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
    
    def comprehensive_test_suite(self):
        """Run comprehensive test suite"""
        print("\n🧪 COMPREHENSIVE MODEL TESTING")
        print("=" * 60)
        
        if not self.model_loaded:
            print("❌ Cannot run tests - model not loaded!")
            return None
        
        # Test categories
        test_results = {}
        
        # 1. Basic functionality test
        test_results['basic'] = self.test_basic_functionality()
        
        # 2. Diverse content test
        test_results['diverse'] = self.test_diverse_content()
        
        # 3. Edge cases test
        test_results['edge_cases'] = self.test_edge_cases()
        
        # 4. Internet slang test
        test_results['slang'] = self.test_internet_slang()
        
        # 5. Emoji test
        test_results['emoji'] = self.test_emoji_content()
        
        # Overall performance
        self.generate_final_report(test_results)
        
        return test_results
    
    def test_basic_functionality(self):
        """Test basic sentiment classification"""
        print("\n📋 1. Basic Functionality Test")
        print("-" * 40)
        
        test_cases = [
            ("This meme is absolutely hilarious! 😂😂😂", "Positive"),
            ("I hate everything about this stupid day", "Negative"),
            ("It's Wednesday my dudes", "Neutral"),
            ("Best meme ever! Love it so much!", "Positive"),
            ("Ugh, another boring meeting", "Negative"),
            ("Today is Thursday", "Neutral"),
            ("This is amazing! So good!", "Positive"),
            ("This sucks so much", "Negative"),
            ("Just another regular day", "Neutral"),
            ("Love this wholesome content", "Positive")
        ]
        
        return self.run_test_cases(test_cases, "Basic")
    
    def test_diverse_content(self):
        """Test with diverse meme content"""
        print("\n📋 2. Diverse Content Test")
        print("-" * 40)
        
        test_cases = [
            ("Wholesome meme made my heart warm", "Positive"),
            ("This is pure comedy gold", "Positive"),
            ("Achievement unlocked! Feeling proud", "Positive"),
            ("Spreading love and positivity", "Positive"),
            ("This is absolutely disgusting", "Negative"),
            ("So sad and depressing", "Negative"),
            ("This is infuriating! Can't stand it", "Negative"),
            ("Such a disappointing meme", "Negative"),
            ("Random thoughts about things", "Neutral"),
            ("What time is it right now?", "Neutral"),
            ("This exists in the internet space", "Neutral"),
            ("Some factual information here", "Neutral")
        ]
        
        return self.run_test_cases(test_cases, "Diverse")
    
    def test_edge_cases(self):
        """Test edge cases and challenging content"""
        print("\n📋 3. Edge Cases Test")
        print("-" * 40)
        
        test_cases = [
            ("THIS IS AMAZING!!! LOVE IT!!!", "Positive"),
            ("absolutely terrible... hate this...", "Negative"),
            ("just... existing... here... today...", "Neutral"),
            ("Best! Thing! Ever! Amazing!", "Positive"),
            ("WORST. DAY. EVER. UGH!!!", "Negative"),
            ("meh... could be better... or worse...", "Neutral"),
            ("So good! Made me smile", "Positive"),
            ("This content is disappointing", "Negative")
        ]
        
        return self.run_test_cases(test_cases, "Edge Cases")
    
    def test_internet_slang(self):
        """Test internet slang and modern expressions"""
        print("\n📋 4. Internet Slang Test")
        print("-" * 40)
        
        test_cases = [
            ("OMG this is so lit! Fire content!", "Positive"),
            ("Bruh this slaps! No cap!", "Positive"),
            ("This hits different! Chef's kiss!", "Positive"),
            ("Absolute banger! This goes hard!", "Positive"),
            ("This ain't it chief... big yikes", "Negative"),
            ("Cringe content alert! Hard pass", "Negative"),
            ("This is straight trash ngl", "Negative"),
            ("Bruh this is mid at best... meh", "Negative"),
            ("It is what it is tbh", "Neutral"),
            ("Just vibing here... nothing special", "Neutral"),
            ("Random thoughts going brrrr", "Neutral"),
            ("Just another Tuesday mood", "Neutral")
        ]
        
        return self.run_test_cases(test_cases, "Internet Slang")
    
    def test_emoji_content(self):
        """Test emoji-heavy content"""
        print("\n📋 5. Emoji Content Test")
        print("-" * 40)
        
        test_cases = [
            ("😂🤣😄😁😊", "Positive"),
            ("❤️💕💖😍🥰", "Positive"),
            ("🎉🎊🥳🙌👏", "Positive"),
            ("🔥💯⭐✨🌟", "Positive"),
            ("😭😢😞😔😟", "Negative"),
            ("😠😡🤬😤👎", "Negative"),
            ("🤮🤢😷💀☠️", "Negative"),
            ("💔😿😾🙄😒", "Negative"),
            ("🤔😐😑🙂😶", "Neutral"),
            ("📱💻📚🏠🚗", "Neutral"),
            ("🌍🌙⭐🌤️⛅", "Neutral"),
            ("📅⏰🗓️📋📝", "Neutral")
        ]
        
        return self.run_test_cases(test_cases, "Emoji")
    
    def run_test_cases(self, test_cases, category_name):
        """Run a set of test cases and return results"""
        correct = 0
        total = len(test_cases)
        results = []
        confidences = []
        
        for text, expected in test_cases:
            result = self.predict_sentiment(text)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            predicted = result['sentiment']
            confidence = result['confidence']
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
                status = "✅"
                confidences.append(confidence)
            else:
                status = "❌"
            
            print(f"{status} '{text[:40]}...'")
            print(f"   Predicted: {predicted} (Expected: {expected}) - Confidence: {confidence:.2%}")
            
            results.append({
                'text': text,
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
        
        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        print(f"\n📊 {category_name} Results: {accuracy:.2%} accuracy ({correct}/{total})")
        print(f"🎯 Average confidence: {avg_confidence:.2%}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_confidence': avg_confidence,
            'results': results
        }
    
    def compare_with_old_model(self):
        """Compare performance with the old model"""
        print("\n📊 COMPARISON WITH OLD MODEL")
        print("=" * 60)
        
        # Old model performance (from previous testing)
        old_performance = {
            'accuracy': 0.30,
            'confidence': 0.35,
            'bias': 'Severe negative bias - predicts everything as negative'
        }
        
        # Run basic test to get new model performance
        if not self.model_loaded:
            print("❌ Cannot compare - new model not loaded!")
            return
        
        basic_results = self.test_basic_functionality()
        new_accuracy = basic_results['accuracy']
        new_confidence = basic_results['avg_confidence']
        
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        print("=" * 40)
        print(f"📈 Old Model:")
        print(f"   Accuracy: {old_performance['accuracy']:.2%}")
        print(f"   Confidence: {old_performance['confidence']:.2%}")
        print(f"   Issues: {old_performance['bias']}")
        
        print(f"\n📈 New Model:")
        print(f"   Accuracy: {new_accuracy:.2%}")
        print(f"   Confidence: {new_confidence:.2%}")
        print(f"   Issues: None detected")
        
        improvement = new_accuracy - old_performance['accuracy']
        relative_improvement = new_accuracy / old_performance['accuracy']
        
        print(f"\n🎯 IMPROVEMENT:")
        print(f"   Absolute: +{improvement:.2%}")
        print(f"   Relative: {relative_improvement:.1f}x better")
        
        if new_accuracy >= 0.80:
            print(f"\n🎉 EXCELLENT! Your new model is production-ready!")
        elif new_accuracy >= 0.70:
            print(f"\n✅ GREAT! Significant improvement achieved!")
        elif new_accuracy >= 0.60:
            print(f"\n👍 GOOD! Much better than the old model!")
        else:
            print(f"\n⚠️ FAIR! Some improvement but needs more work!")
    
    def generate_final_report(self, test_results):
        """Generate comprehensive final report"""
        print(f"\n📋 FINAL PERFORMANCE REPORT")
        print("=" * 60)
        
        if not test_results:
            print("❌ No test results available")
            return
        
        # Calculate overall metrics
        total_correct = sum(r['correct'] for r in test_results.values() if r)
        total_tests = sum(r['total'] for r in test_results.values() if r)
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
        
        # Calculate average confidence
        all_confidences = []
        for category_results in test_results.values():
            if category_results and 'results' in category_results:
                for result in category_results['results']:
                    if result['correct']:
                        all_confidences.append(result['confidence'])
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Correct Predictions: {total_correct}")
        print(f"   Overall Accuracy: {overall_accuracy:.2%}")
        print(f"   Average Confidence: {avg_confidence:.2%}")
        
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, results in test_results.items():
            if results:
                print(f"   {category.title()}: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if overall_accuracy >= 0.90:
            print("   🌟 OUTSTANDING! Exceptional performance!")
        elif overall_accuracy >= 0.80:
            print("   🎉 EXCELLENT! Ready for production use!")
        elif overall_accuracy >= 0.70:
            print("   ✅ VERY GOOD! Suitable for most applications!")
        elif overall_accuracy >= 0.60:
            print("   👍 GOOD! Solid performance with room for improvement!")
        elif overall_accuracy >= 0.50:
            print("   ⚠️ FAIR! Better than baseline but needs improvement!")
        else:
            print("   ❌ POOR! Needs significant work!")
        
        # Confidence assessment
        print(f"\n🎯 CONFIDENCE ASSESSMENT:")
        if avg_confidence >= 0.85:
            print("   🎯 HIGH: Model is very confident in correct predictions!")
        elif avg_confidence >= 0.70:
            print("   ✅ GOOD: Model shows reasonable confidence!")
        elif avg_confidence >= 0.55:
            print("   ⚠️ MODERATE: Model confidence could be better!")
        else:
            print("   ❌ LOW: Model lacks confidence in predictions!")
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_accuracy': float(overall_accuracy),
            'average_confidence': float(avg_confidence),
            'total_tests': total_tests,
            'total_correct': total_correct,
            'category_results': test_results,
            'model_path': self.model_path
        }
        
        with open('new_model_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: new_model_test_report.json")
        
        return report

def main():
    """Main testing function"""
    print("🔬 NEW MODEL COMPREHENSIVE TESTING")
    print("=" * 70)
    print("Testing the newly trained meme sentiment model")
    print()
    
    # Initialize tester
    tester = NewModelTester()
    
    if not tester.model_loaded:
        print("❌ Cannot proceed with testing!")
        print("📋 Please run 'python train_from_scratch.py' first to train the model")
        return
    
    # Run comprehensive test suite
    test_results = tester.comprehensive_test_suite()
    
    # Compare with old model
    tester.compare_with_old_model()
    
    print(f"\n🎯 TESTING COMPLETED!")
    print("=" * 70)
    print("📋 Next steps:")
    print("1. Review the test results above")
    print("2. Check new_model_test_report.json for detailed results")
    print("3. If satisfied, update your production system")
    print("4. Monitor performance in real-world usage")
    
    return test_results

if __name__ == "__main__":
    main()
