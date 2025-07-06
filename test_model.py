#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite for Meme Sentiment Analyzer
Tests model performance, accuracy, and provides detailed evaluation metrics
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from meme_sentiment_analyzer import MemeSentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    """Comprehensive model testing and evaluation"""
    
    def __init__(self, model_path=None):
        """Initialize model tester"""
        self.analyzer = MemeSentimentAnalyzer(model_path)
        self.sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.label_to_id = {"Negative": 0, "Neutral": 1, "Positive": 2}
        
    def test_basic_functionality(self):
        """Test basic functionality with sample inputs"""
        print("🧪 Testing Basic Functionality")
        print("=" * 50)
        
        test_cases = [
            {"text": "This meme is hilarious! 😂", "expected": "Positive"},
            {"text": "I hate Mondays so much 😭", "expected": "Negative"},
            {"text": "It's Wednesday my dudes", "expected": "Neutral"},
            {"text": "This is the worst thing ever", "expected": "Negative"},
            {"text": "I love this wholesome content ❤️", "expected": "Positive"},
            {"text": "Meh, it's okay I guess", "expected": "Neutral"},
            {"text": "Amazing! This made my day!", "expected": "Positive"},
            {"text": "Ugh, another boring meeting", "expected": "Negative"},
            {"text": "Today is Thursday", "expected": "Neutral"},
            {"text": "Best meme ever! So funny!", "expected": "Positive"}
        ]
        
        correct = 0
        total = len(test_cases)
        results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {case['text']}")
            result = self.analyzer.analyze_text_sentiment(case['text'])
            
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
                'correct': is_correct
            })
        
        accuracy = correct / total
        print(f"\n📊 Basic Test Results:")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return results
    
    def test_with_dataset(self, csv_path="dataset.csv"):
        """Test model with existing dataset"""
        print(f"\n📊 Testing with Dataset: {csv_path}")
        print("=" * 50)
        
        if not os.path.exists(csv_path):
            print(f"❌ Dataset file not found: {csv_path}")
            return None
        
        try:
            # Load dataset
            df = pd.read_csv(csv_path)
            print(f"📄 Loaded dataset with {len(df)} samples")
            
            # Check required columns
            if 'text' not in df.columns or 'sentiment' not in df.columns:
                print("❌ Dataset must have 'text' and 'sentiment' columns")
                return None
            
            # Sample data for testing (to avoid long processing times)
            test_size = min(100, len(df))
            test_df = df.sample(n=test_size, random_state=42)
            print(f"🎯 Testing with {test_size} samples")
            
            predictions = []
            true_labels = []
            confidences = []
            
            print("🔄 Analyzing samples...")
            for idx, row in test_df.iterrows():
                text = str(row['text'])
                true_sentiment = str(row['sentiment']).strip()
                
                # Standardize sentiment labels
                if true_sentiment.lower() in ['positive', 'pos', '2']:
                    true_sentiment = "Positive"
                elif true_sentiment.lower() in ['negative', 'neg', '0']:
                    true_sentiment = "Negative"
                elif true_sentiment.lower() in ['neutral', 'neu', '1']:
                    true_sentiment = "Neutral"
                
                result = self.analyzer.analyze_text_sentiment(text)
                
                if 'error' not in result:
                    predictions.append(result['sentiment'])
                    true_labels.append(true_sentiment)
                    confidences.append(result['confidence'])
                    
                    # Print progress every 20 samples
                    if len(predictions) % 20 == 0:
                        print(f"  Processed {len(predictions)}/{test_size} samples...")
            
            if len(predictions) == 0:
                print("❌ No valid predictions made")
                return None
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, average='weighted'
            )
            
            print(f"\n📈 Dataset Test Results:")
            print(f"Samples tested: {len(predictions)}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            print(f"Average Confidence: {np.mean(confidences):.3f}")
            
            # Detailed classification report
            print(f"\n📋 Detailed Classification Report:")
            print(classification_report(true_labels, predictions))
            
            # Confusion Matrix
            cm = confusion_matrix(true_labels, predictions, labels=["Negative", "Neutral", "Positive"])
            self.plot_confusion_matrix(cm, ["Negative", "Neutral", "Positive"])
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions,
                'true_labels': true_labels,
                'confidences': confidences
            }
            
        except Exception as e:
            print(f"❌ Error testing with dataset: {e}")
            return None
    
    def test_image_analysis(self, image_dir="meme_dataset"):
        """Test image analysis functionality"""
        print(f"\n🖼️ Testing Image Analysis: {image_dir}")
        print("=" * 50)
        
        if not os.path.exists(image_dir):
            print(f"❌ Image directory not found: {image_dir}")
            return None
        
        # Find image files
        image_files = []
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_files.append(os.path.join(image_dir, file))
        
        if not image_files:
            print("❌ No image files found")
            return None
        
        print(f"📁 Found {len(image_files)} images")
        
        # Test sample of images
        test_images = image_files[:10]  # Test first 10 images
        results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n🔍 Testing image {i}: {os.path.basename(image_path)}")
            
            result = self.analyzer.analyze_image_sentiment(image_path)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            extracted_text = result.get('extracted_text', '')
            sentiment = result.get('sentiment', 'Unknown')
            confidence = result.get('confidence', 0)
            
            print(f"📝 Extracted text: {extracted_text[:100]}...")
            print(f"📈 Sentiment: {sentiment} (Confidence: {confidence:.2%})")
            
            results.append({
                'image': os.path.basename(image_path),
                'extracted_text': extracted_text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        print(f"\n📊 Image Analysis Summary:")
        print(f"Images processed: {len(results)}")
        
        if results:
            sentiments = [r['sentiment'] for r in results]
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
            
            for sentiment, count in sentiment_counts.items():
                print(f"{sentiment}: {count} images")
        
        return results
    
    def test_edge_cases(self):
        """Test edge cases and robustness"""
        print("\n🔧 Testing Edge Cases")
        print("=" * 50)
        
        edge_cases = [
            {"text": "", "description": "Empty string"},
            {"text": "   ", "description": "Whitespace only"},
            {"text": "😂😭😍😡🤔", "description": "Emojis only"},
            {"text": "AAAAAAHHHHHHH!!!", "description": "Screaming"},
            {"text": "a" * 1000, "description": "Very long text"},
            {"text": "lol wut dis meme iz cray cray", "description": "Internet slang"},
            {"text": "🙃🔥💯✨🌟⭐", "description": "Mixed emojis"},
            {"text": "123 456 789", "description": "Numbers only"},
            {"text": "This is fine. 🔥🐕", "description": "Sarcastic with context"},
        ]
        
        results = []
        for case in edge_cases:
            print(f"\n🧪 Testing: {case['description']}")
            print(f"Input: '{case['text'][:50]}...' ")
            
            try:
                result = self.analyzer.analyze_text_sentiment(case['text'])
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    status = "Error"
                else:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    print(f"✅ Sentiment: {sentiment} (Confidence: {confidence:.2%})")
                    status = "Success"
                
                results.append({
                    'description': case['description'],
                    'text': case['text'],
                    'status': status,
                    'result': result
                })
                
            except Exception as e:
                print(f"❌ Exception: {e}")
                results.append({
                    'description': case['description'],
                    'text': case['text'],
                    'status': "Exception",
                    'result': str(e)
                })
        
        success_rate = sum(1 for r in results if r['status'] == 'Success') / len(results)
        print(f"\n📊 Edge Case Results:")
        print(f"Success rate: {success_rate:.2%}")
        
        return results
    
    def benchmark_performance(self, num_samples=50):
        """Benchmark model performance and speed"""
        print(f"\n⏱️ Benchmarking Performance ({num_samples} samples)")
        print("=" * 50)
        
        # Generate test samples
        test_texts = [
            "This meme is amazing!",
            "I hate everything about this",
            "It's just another day",
            "Best thing ever! So funny!",
            "Worst day of my life",
            "Wednesday is here",
            "Love this so much ❤️",
            "Ugh, terrible",
            "Random meme text",
            "Cool story bro"
        ] * (num_samples // 10 + 1)
        
        test_texts = test_texts[:num_samples]
        
        import time
        start_time = time.time()
        
        results = []
        for i, text in enumerate(test_texts):
            iter_start = time.time()
            result = self.analyzer.analyze_text_sentiment(text)
            iter_time = time.time() - iter_start
            
            results.append({
                'text': text,
                'processing_time': iter_time,
                'result': result
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples...")
        
        total_time = time.time() - start_time
        avg_time = total_time / num_samples
        
        print(f"\n⚡ Performance Results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per sample: {avg_time:.3f} seconds")
        print(f"Throughput: {num_samples / total_time:.1f} samples/second")
        
        # Plot processing times
        times = [r['processing_time'] for r in results]
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(times, bins=20, alpha=0.7)
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.plot(times)
        plt.title('Processing Time Over Samples')
        plt.xlabel('Sample Number')
        plt.ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_test_report(self, output_file="test_report.json"):
        """Generate comprehensive test report"""
        print(f"\n📝 Generating Comprehensive Test Report")
        print("=" * 50)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_path': self.analyzer.model_path,
            'tests': {}
        }
        
        # Run all tests
        try:
            print("Running basic functionality tests...")
            report['tests']['basic_functionality'] = self.test_basic_functionality()
        except Exception as e:
            print(f"Basic functionality test failed: {e}")
        
        try:
            print("Running dataset tests...")
            report['tests']['dataset_evaluation'] = self.test_with_dataset()
        except Exception as e:
            print(f"Dataset test failed: {e}")
        
        try:
            print("Running image analysis tests...")
            report['tests']['image_analysis'] = self.test_image_analysis()
        except Exception as e:
            print(f"Image analysis test failed: {e}")
        
        try:
            print("Running edge case tests...")
            report['tests']['edge_cases'] = self.test_edge_cases()
        except Exception as e:
            print(f"Edge case test failed: {e}")
        
        try:
            print("Running performance benchmark...")
            report['tests']['performance'] = self.benchmark_performance()
        except Exception as e:
            print(f"Performance benchmark failed: {e}")
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Test report saved to: {output_file}")
        return report

def main():
    """Run comprehensive model testing"""
    print("🧪 Meme Sentiment Model Testing Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = ModelTester()
    
    # Menu for different test types
    while True:
        print("\n🎯 Choose a test to run:")
        print("1. 🔧 Basic functionality test")
        print("2. 📊 Dataset evaluation")
        print("3. 🖼️ Image analysis test") 
        print("4. 🧪 Edge cases test")
        print("5. ⚡ Performance benchmark")
        print("6. 📝 Generate full test report")
        print("7. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            tester.test_basic_functionality()
        elif choice == '2':
            tester.test_with_dataset()
        elif choice == '3':
            tester.test_image_analysis()
        elif choice == '4':
            tester.test_edge_cases()
        elif choice == '5':
            tester.benchmark_performance()
        elif choice == '6':
            tester.generate_test_report()
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main()
