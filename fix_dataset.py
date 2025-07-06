#!/usr/bin/env python3
"""
Fix dataset format for proper testing
"""

import pandas as pd
import os

def fix_dataset():
    """Fix the dataset format"""
    print("🔧 Fixing Dataset Format")
    print("=" * 40)
    
    # Load original dataset
    df = pd.read_csv("dataset.csv")
    print(f"📄 Original dataset: {len(df)} rows")
    print(f"📋 Original columns: {list(df.columns)}")
    
    # Create new dataset with proper format
    fixed_df = pd.DataFrame()
    
    # Map columns
    fixed_df['text'] = df['extracted_text']
    
    # Standardize sentiment labels
    sentiment_mapping = {
        'positive': 'Positive',
        'negative': 'Negative', 
        'neutral': 'Neutral'
    }
    
    fixed_df['sentiment'] = df['label'].map(sentiment_mapping)
    
    # Remove rows with no text
    fixed_df = fixed_df[fixed_df['text'].notna()]
    fixed_df = fixed_df[fixed_df['text'] != 'No Text']
    fixed_df = fixed_df[fixed_df['text'].str.strip() != '']
    
    # Save fixed dataset
    fixed_df.to_csv("dataset_fixed.csv", index=False)
    
    print(f"✅ Fixed dataset saved: {len(fixed_df)} rows")
    print(f"📋 New columns: {list(fixed_df.columns)}")
    
    # Show sentiment distribution
    sentiment_counts = fixed_df['sentiment'].value_counts()
    print(f"\n📈 Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} samples")
    
    # Show sample data
    print(f"\n📝 Sample Data:")
    for idx, row in fixed_df.head().iterrows():
        print(f"  Text: {row['text'][:50]}...")
        print(f"  Sentiment: {row['sentiment']}\n")
    
    return fixed_df

def create_additional_test_data():
    """Create additional test data for better evaluation"""
    print("\n📊 Creating Additional Test Data")
    print("=" * 40)
    
    additional_data = [
        # Positive examples
        {"text": "This meme is absolutely hilarious! 😂😂😂", "sentiment": "Positive"},
        {"text": "Best meme ever! Love it so much!", "sentiment": "Positive"},
        {"text": "This made my day! So wholesome ❤️", "sentiment": "Positive"},
        {"text": "Amazing content! Keep it up!", "sentiment": "Positive"},
        {"text": "LOL this is so funny! Can't stop laughing", "sentiment": "Positive"},
        {"text": "Wholesome and heartwarming meme", "sentiment": "Positive"},
        {"text": "This meme brings me joy", "sentiment": "Positive"},
        {"text": "Perfect meme for today! Love it!", "sentiment": "Positive"},
        {"text": "This is comedy gold! 😂", "sentiment": "Positive"},
        {"text": "Such a clever and funny meme", "sentiment": "Positive"},
        
        # Negative examples
        {"text": "I hate everything about this", "sentiment": "Negative"},
        {"text": "This is the worst thing ever", "sentiment": "Negative"},
        {"text": "Ugh, another boring Monday", "sentiment": "Negative"},
        {"text": "This meme is terrible and unfunny", "sentiment": "Negative"},
        {"text": "I'm so tired of this nonsense", "sentiment": "Negative"},
        {"text": "This makes me angry and frustrated", "sentiment": "Negative"},
        {"text": "What a horrible day", "sentiment": "Negative"},
        {"text": "This content is disappointing", "sentiment": "Negative"},
        {"text": "I can't stand this anymore", "sentiment": "Negative"},
        {"text": "This is so annoying and stupid", "sentiment": "Negative"},
        
        # Neutral examples
        {"text": "It's Wednesday my dudes", "sentiment": "Neutral"},
        {"text": "Today is Thursday", "sentiment": "Neutral"},
        {"text": "The weather is cloudy", "sentiment": "Neutral"},
        {"text": "I went to the store", "sentiment": "Neutral"},
        {"text": "This is a meme", "sentiment": "Neutral"},
        {"text": "Some text about things", "sentiment": "Neutral"},
        {"text": "Random meme content here", "sentiment": "Neutral"},
        {"text": "It is what it is", "sentiment": "Neutral"},
        {"text": "Things happen sometimes", "sentiment": "Neutral"},
        {"text": "This exists I guess", "sentiment": "Neutral"},
    ]
    
    additional_df = pd.DataFrame(additional_data)
    additional_df.to_csv("additional_test_data.csv", index=False)
    
    print(f"✅ Created {len(additional_df)} additional test samples")
    
    return additional_df

def combine_datasets():
    """Combine fixed dataset with additional test data"""
    print("\n🔗 Combining Datasets")
    print("=" * 40)
    
    # Load datasets
    fixed_df = pd.read_csv("dataset_fixed.csv")
    additional_df = pd.read_csv("additional_test_data.csv")
    
    # Combine
    combined_df = pd.concat([fixed_df, additional_df], ignore_index=True)
    
    # Save combined dataset
    combined_df.to_csv("dataset_combined.csv", index=False)
    
    print(f"✅ Combined dataset: {len(combined_df)} rows")
    
    # Show final distribution
    sentiment_counts = combined_df['sentiment'].value_counts()
    print(f"\n📈 Final Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} samples ({count/len(combined_df)*100:.1f}%)")
    
    return combined_df

def main():
    """Fix dataset and prepare for testing"""
    print("🛠️ Dataset Preparation for Testing")
    print("=" * 50)
    
    # Fix original dataset
    fixed_df = fix_dataset()
    
    # Create additional test data
    additional_df = create_additional_test_data()
    
    # Combine datasets
    combined_df = combine_datasets()
    
    print(f"\n✅ Dataset preparation completed!")
    print(f"📁 Files created:")
    print(f"  - dataset_fixed.csv: {len(fixed_df)} samples (original data cleaned)")
    print(f"  - additional_test_data.csv: {len(additional_df)} samples (new test data)")
    print(f"  - dataset_combined.csv: {len(combined_df)} samples (all data combined)")
    
    print(f"\n📋 Next steps:")
    print(f"1. Run 'python test_model.py' with the fixed dataset")
    print(f"2. Test with different datasets to see model performance")

if __name__ == "__main__":
    main()
