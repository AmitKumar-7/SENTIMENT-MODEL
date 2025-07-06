#!/usr/bin/env python3
"""
Comprehensive test script for the saved sentiment model
Tests model loading, inference, and basic functionality
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
import traceback

class ModelTester:
    def __init__(self, model_path="./production_model_v2"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
    def load_model(self):
        """Load the saved model and tokenizer"""
        try:
            print(f"🔄 Loading model from: {self.model_path}")
            
            # Check if model files exist
            required_files = ['model.safetensors', 'config.json', 'tokenizer_config.json', 'vocab.txt']
            for file in required_files:
                file_path = os.path.join(self.model_path, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("✅ Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            print("✅ Model loaded successfully")
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            print("✅ Pipeline created successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_basic_inference(self):
        """Test basic model inference"""
        print("\n🧪 Testing Basic Inference...")
        
        test_texts = [
            "This meme is absolutely hilarious! 😂",
            "I hate this stupid meme",
            "This is just a regular meme",
            "Best meme ever! Love it!",
            "This meme is okay, nothing special",
            "Worst meme I've ever seen"
        ]
        
        results = []
        
        for i, text in enumerate(test_texts, 1):
            try:
                # Get prediction
                prediction = self.pipeline(text)
                
                # Handle different pipeline output formats
                if isinstance(prediction, list) and len(prediction) > 0:
                    if isinstance(prediction[0], dict):
                        # Single prediction result
                        pred_dict = prediction[0]
                        predicted_label = pred_dict.get('label', 'UNKNOWN')
                        confidence = pred_dict.get('score', 0.0)
                        scores = {predicted_label: confidence}
                    elif isinstance(prediction[0], list):
                        # Multiple scores per prediction
                        pred_scores = prediction[0]
                        scores = {item['label']: item['score'] for item in pred_scores}
                        predicted_label = max(scores, key=scores.get)
                        confidence = max(scores.values())
                    else:
                        # Fallback
                        predicted_label = str(prediction[0])
                        confidence = 1.0
                        scores = {predicted_label: confidence}
                else:
                    # Direct dictionary result
                    predicted_label = prediction.get('label', 'UNKNOWN')
                    confidence = prediction.get('score', 0.0)
                    scores = {predicted_label: confidence}
                
                # Map to sentiment
                if predicted_label in ['LABEL_0', 'NEGATIVE']:
                    sentiment = "Negative"
                elif predicted_label in ['LABEL_1', 'NEUTRAL']:
                    sentiment = "Neutral"
                elif predicted_label in ['LABEL_2', 'POSITIVE']:
                    sentiment = "Positive"
                else:
                    sentiment = predicted_label
                
                result = {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "all_scores": scores
                }
                
                results.append(result)
                
                print(f"  {i}. Text: '{text[:50]}...' → {sentiment} ({confidence:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error processing text {i}: {str(e)}")
                results.append({"text": text, "error": str(e)})
        
        return results
    
    def test_edge_cases(self):
        """Test edge cases and special inputs"""
        print("\n🔍 Testing Edge Cases...")
        
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "😂😂😂",  # Only emojis
            "a" * 1000,  # Very long text
            "Mixed feelings about this meme 😕 but also 😊",  # Mixed sentiment
            "CAPS LOCK MEME TEXT!!!",  # All caps
            "meme with numbers 123 and symbols !@#$%",  # Special characters
        ]
        
        results = []
        
        for i, text in enumerate(edge_cases, 1):
            try:
                if len(text) > 100:
                    display_text = text[:50] + "..." + text[-10:]
                else:
                    display_text = text
                
                prediction = self.pipeline(text)
                
                # Handle different pipeline output formats
                if isinstance(prediction, list) and len(prediction) > 0:
                    if isinstance(prediction[0], dict):
                        # Single prediction result
                        pred_dict = prediction[0]
                        predicted_label = pred_dict.get('label', 'UNKNOWN')
                        confidence = pred_dict.get('score', 0.0)
                        scores = {predicted_label: confidence}
                    elif isinstance(prediction[0], list):
                        # Multiple scores per prediction
                        pred_scores = prediction[0]
                        scores = {item['label']: item['score'] for item in pred_scores}
                        predicted_label = max(scores, key=scores.get)
                        confidence = max(scores.values())
                    else:
                        # Fallback
                        predicted_label = str(prediction[0])
                        confidence = 1.0
                        scores = {predicted_label: confidence}
                else:
                    # Direct dictionary result
                    predicted_label = prediction.get('label', 'UNKNOWN')
                    confidence = prediction.get('score', 0.0)
                    scores = {predicted_label: confidence}
                
                # Map to sentiment
                if predicted_label in ['LABEL_0', 'NEGATIVE']:
                    sentiment = "Negative"
                elif predicted_label in ['LABEL_1', 'NEUTRAL']:
                    sentiment = "Neutral"
                elif predicted_label in ['LABEL_2', 'POSITIVE']:
                    sentiment = "Positive"
                else:
                    sentiment = predicted_label
                
                result = {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "all_scores": scores
                }
                
                results.append(result)
                
                print(f"  {i}. '{display_text}' → {sentiment} ({confidence:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error processing edge case {i}: {str(e)}")
                results.append({"text": text, "error": str(e)})
        
        return results
    
    def test_model_info(self):
        """Test model information and configuration"""
        print("\n📊 Model Information...")
        
        try:
            # Model config
            config = self.model.config
            print(f"  Model Type: {config.model_type}")
            print(f"  Number of Labels: {config.num_labels}")
            print(f"  Hidden Size: {config.hidden_size}")
            print(f"  Attention Heads: {config.num_attention_heads}")
            print(f"  Hidden Layers: {config.num_hidden_layers}")
            
            # Tokenizer info
            print(f"  Vocabulary Size: {len(self.tokenizer.vocab)}")
            print(f"  Max Length: {self.tokenizer.model_max_length}")
            
            # Model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error getting model info: {str(e)}")
            return False
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        print("\n📦 Testing Batch Processing...")
        
        batch_texts = [
            "This meme is funny!",
            "I don't like this meme",
            "Average meme, nothing special",
            "Great meme, love it!",
            "Terrible meme"
        ]
        
        try:
            # Process batch
            batch_results = self.pipeline(batch_texts)
            
            print(f"  Processed {len(batch_texts)} texts in batch")
            
            for i, (text, result) in enumerate(zip(batch_texts, batch_results)):
                scores = {item['label']: item['score'] for item in result}
                predicted_label = max(scores, key=scores.get)
                confidence = max(scores.values())
                
                # Map to sentiment
                if predicted_label in ['LABEL_0', 'NEGATIVE']:
                    sentiment = "Negative"
                elif predicted_label in ['LABEL_1', 'NEUTRAL']:
                    sentiment = "Neutral"
                elif predicted_label in ['LABEL_2', 'POSITIVE']:
                    sentiment = "Positive"
                else:
                    sentiment = predicted_label
                
                print(f"    {i+1}. '{text}' → {sentiment} ({confidence:.3f})")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error in batch processing: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🚀 Starting Comprehensive Model Testing...")
        print("="*60)
        
        # Test results
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "tests": {}
        }
        
        # 1. Load model
        load_success = self.load_model()
        test_results["tests"]["model_loading"] = {"success": load_success}
        
        if not load_success:
            print("❌ Model loading failed. Stopping tests.")
            return test_results
        
        # 2. Model info
        info_success = self.test_model_info()
        test_results["tests"]["model_info"] = {"success": info_success}
        
        # 3. Basic inference
        basic_results = self.test_basic_inference()
        test_results["tests"]["basic_inference"] = {
            "success": len([r for r in basic_results if "error" not in r]) > 0,
            "results": basic_results
        }
        
        # 4. Edge cases
        edge_results = self.test_edge_cases()
        test_results["tests"]["edge_cases"] = {
            "success": len([r for r in edge_results if "error" not in r]) > 0,
            "results": edge_results
        }
        
        # 5. Batch processing
        batch_success = self.test_batch_processing()
        test_results["tests"]["batch_processing"] = {"success": batch_success}
        
        # Summary
        print("\n" + "="*60)
        print("📋 Test Summary:")
        
        total_tests = len(test_results["tests"])
        passed_tests = sum(1 for test in test_results["tests"].values() if test["success"])
        
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("✅ All tests passed! Model is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return test_results
    
    def save_test_report(self, test_results, filename="model_test_report.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            print(f"📄 Test report saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving test report: {str(e)}")
            return False

def main():
    """Main function to run the model tests"""
    print("🎯 Meme Sentiment Model Tester")
    print("="*60)
    
    # Initialize tester
    tester = ModelTester()
    
    # Run all tests
    test_results = tester.run_all_tests()
    
    # Save report
    tester.save_test_report(test_results)
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
