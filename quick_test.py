#!/usr/bin/env python3
"""
Quick Test Script - Basic model validation
"""

import pandas as pd
import os
from meme_sentiment_analyzer import MemeSentimentAnalyzer

def check_dataset():
    """Check dataset structure and content"""
    print("📊 Checking Dataset Structure")
    print("=" * 40)
    
    if not os.path.exists("dataset.csv"):
        print("❌ dataset.csv not found")
        return None
    
    df = pd.read_csv("dataset.csv")
    print(f"✅ Dataset loaded: {len(df)} rows")
    print(f"📋 Columns: {list(df.columns)}")
    
    if 'text' in df.columns and 'sentiment' in df.columns:
        print("✅ Required columns found: 'text' and 'sentiment'")
        
        # Check sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        print(f"\n📈 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} samples")
        
        # Show sample data
        print(f"\n📝 Sample Data:")
        print(df.head())
        
        return df
    else:
        print("❌ Missing required columns: 'text' and 'sentiment'")
        return None

def quick_model_test():
    """Quick model functionality test"""
    print("\n🧪 Quick Model Test")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = MemeSentimentAnalyzer()
    
    # Test cases
    test_cases = [
        "This meme is absolutely hilarious! 😂😂😂",
        "I hate everything about this stupid day",
        "It's Wednesday my dudes",
        "Best meme ever! Love it!",
        "Ugh, another boring meeting"
    ]
    
    print("Testing sample inputs:")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{text}'")
        result = analyzer.analyze_text_sentiment(text)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            sentiment = result['sentiment']
            confidence = result['confidence']
            categories = analyzer.categorize_meme(sentiment, text)
            
            print(f"   📈 Sentiment: {sentiment}")
            print(f"   🎯 Confidence: {confidence:.2%}")
            print(f"   🏷️ Categories: {', '.join(categories[:2])}")

def test_image_sample():
    """Test with a sample image"""
    print("\n🖼️ Testing Image Analysis")
    print("=" * 40)
    
    # Find first image in meme_dataset
    if os.path.exists("meme_dataset"):
        images = [f for f in os.listdir("meme_dataset") if f.lower().endswith(('.jpg', '.png', '.gif'))]
        if images:
            image_path = os.path.join("meme_dataset", images[0])
            print(f"Testing image: {images[0]}")
            
            analyzer = MemeSentimentAnalyzer()
            result = analyzer.analyze_image_sentiment(image_path)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"📝 Extracted text: {result.get('extracted_text', 'None')}")
                print(f"📈 Sentiment: {result.get('sentiment', 'Unknown')}")
                print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
        else:
            print("❌ No images found in meme_dataset")
    else:
        print("❌ meme_dataset directory not found")

def main():
    """Run quick tests"""
    print("🚀 Quick Model Testing")
    print("=" * 50)
    
    # Check dataset
    dataset = check_dataset()
    
    # Test model functionality
    quick_model_test()
    
    # Test image analysis
    test_image_sample()
    
    print("\n✅ Quick test completed!")
    print("\n📋 Next steps:")
    print("1. Run 'python test_model.py' for comprehensive testing")
    print("2. Run 'python meme_cli.py \"your text\"' to test CLI")
    print("3. Run 'python web_app.py' to start web interface")

if __name__ == "__main__":
    main()
