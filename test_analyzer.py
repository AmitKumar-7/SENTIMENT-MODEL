#!/usr/bin/env python3
"""
Test script for Meme Sentiment Analyzer
Demonstrates functionality with sample inputs
"""

import json
import os
from meme_sentiment_analyzer import MemeSentimentAnalyzer

def test_analyzer():
    """Test the meme sentiment analyzer with various inputs"""
    
    print("🎯 Testing Meme Sentiment Analyzer")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        print("📊 Initializing analyzer...")
        analyzer = MemeSentimentAnalyzer()
        print("✅ Analyzer initialized successfully!")
        print()
        
        # Test cases
        test_cases = [
            {
                "input": "This meme is absolutely hilarious! I can't stop laughing 😂😂😂",
                "type": "text",
                "description": "Positive funny text"
            },
            {
                "input": "When you realize it's Monday tomorrow... 😭",
                "type": "text", 
                "description": "Negative Monday blues"
            },
            {
                "input": "It's Wednesday my dudes",
                "type": "text",
                "description": "Neutral observational"
            },
            {
                "input": "This wholesome meme made my day! So sweet and heartwarming ❤️",
                "type": "text",
                "description": "Positive wholesome"
            },
            {
                "input": "Oh great, another meeting that could have been an email... 🙄",
                "type": "text",
                "description": "Negative sarcastic"
            }
        ]
        
        # Run tests
        for i, test_case in enumerate(test_cases, 1):
            print(f"🧪 Test {i}: {test_case['description']}")
            print(f"📝 Input: {test_case['input']}")
            print("🔄 Analyzing...")
            
            try:
                result = analyzer.analyze_meme(test_case['input'], test_case['type'])
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    # Display results
                    sentiment = result.get('sentiment', 'Unknown')
                    confidence = result.get('confidence', 0)
                    categories = result.get('meme_categories', [])
                    
                    # Emoji for sentiment
                    sentiment_emoji = {
                        'Positive': '😊',
                        'Negative': '😞', 
                        'Neutral': '😐'
                    }.get(sentiment, '🤔')
                    
                    print(f"📈 Sentiment: {sentiment_emoji} {sentiment}")
                    print(f"🎯 Confidence: {confidence:.2%}")
                    print(f"🏷️ Categories: {', '.join(categories) if categories else 'None'}")
                    
                    if 'all_scores' in result:
                        print("📊 Detailed Scores:")
                        for sent, score in result['all_scores'].items():
                            print(f"   {sent}: {score:.2%}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
            
            print("-" * 30)
            print()
        
        # Test with sample image if available
        sample_images = []
        for root, dirs, files in os.walk('meme_dataset'):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    sample_images.append(os.path.join(root, file))
                    break  # Just test one image
            if sample_images:
                break
        
        if sample_images:
            print("🖼️ Testing image analysis...")
            image_path = sample_images[0]
            print(f"📁 Image: {image_path}")
            
            try:
                result = analyzer.analyze_meme(image_path, "image")
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"📈 Sentiment: {result.get('sentiment', 'Unknown')}")
                    print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
                    print(f"📝 Extracted Text: {result.get('extracted_text', 'None')}")
            except Exception as e:
                print(f"❌ Image test failed: {e}")
        else:
            print("ℹ️ No sample images found for testing")
        
        print()
        print("✅ Testing completed!")
        print("🚀 You can now use the analyzer with:")
        print("   - Python API: from meme_sentiment_analyzer import MemeSentimentAnalyzer")
        print("   - CLI: python meme_cli.py 'Your meme text here'")
        print("   - Web: python web_app.py")
        
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        print("💡 Make sure you have:")
        print("   1. Installed all requirements: pip install -r requirements.txt")
        print("   2. A trained model in the production_model_v2/ directory")
        print("   3. Proper configuration in your .env file")

if __name__ == "__main__":
    test_analyzer()
