#!/usr/bin/env python3
"""
Test script to analyze sarcastic meme classification issue
"""

from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer
import json

def test_sarcasm_classification():
    """Test how the model handles sarcastic content"""
    
    print('🔍 Testing Sarcastic Meme Classification Issue')
    print('=' * 50)

    analyzer = AdvancedMultimodalAnalyzer()

    # Test sarcastic examples
    sarcastic_examples = [
        {
            "text": "Oh great, another Monday meme 🙄",
            "expected": "sarcastic (could be negative or neutral)",
            "context": "Rolling eyes emoji indicates sarcasm"
        },
        {
            "text": "Wow, such original content 😏",
            "expected": "sarcastic (could be negative)",
            "context": "Smirking emoji indicates sarcasm"
        },
        {
            "text": "Yeah, because that always works out perfectly",
            "expected": "sarcastic (negative undertone)",
            "context": "Ironic statement"
        },
        {
            "text": "Oh sure, let me just drop everything and work on your problem",
            "expected": "sarcastic (negative/frustrated)",
            "context": "Sarcastic agreement"
        },
        {
            "text": "Thanks for the reminder, as if I could forget 🙃",
            "expected": "sarcastic (could be positive or negative)",
            "context": "Upside-down face indicates sarcasm"
        },
        {
            "text": "Perfect timing, just what I needed today",
            "expected": "sarcastic (negative)",
            "context": "Ironic statement about bad timing"
        }
    ]

    print('📝 Testing Sarcastic Text Examples:')
    print('-' * 40)

    results = []
    for i, example in enumerate(sarcastic_examples, 1):
        text = example["text"]
        expected = example["expected"]
        context = example["context"]
        
        result = analyzer.analyze_meme(text, 'text')
        sentiment = result.get('sentiment', 'Unknown')
        confidence = result.get('confidence', 0)
        
        print(f'{i}. "{text}"')
        print(f'   Expected: {expected}')
        print(f'   Actual: {sentiment} ({confidence:.1%})')
        print(f'   Context: {context}')
        
        # Analyze if classification makes sense
        is_reasonable = True
        if sentiment == "Negative":
            reason = "✅ Reasonable - sarcasm often has negative undertones"
        elif sentiment == "Neutral":
            reason = "⚠️ Debatable - could be neutral if sarcasm is mild"
        else:  # Positive
            reason = "❌ Problematic - clearly sarcastic, not genuinely positive"
        
        print(f'   Analysis: {reason}')
        
        results.append({
            "text": text,
            "expected": expected,
            "actual": sentiment,
            "confidence": confidence,
            "is_reasonable": is_reasonable,
            "context": context
        })
        print()

    # Test some genuinely positive examples for comparison
    print('\n📝 Testing Genuinely Positive Examples for Comparison:')
    print('-' * 40)
    
    positive_examples = [
        "This meme actually made me laugh out loud!",
        "I love this wholesome content ❤️",
        "This is genuinely funny and clever",
        "Such a heartwarming meme, thanks for sharing!"
    ]
    
    for i, text in enumerate(positive_examples, 1):
        result = analyzer.analyze_meme(text, 'text')
        sentiment = result.get('sentiment', 'Unknown')
        confidence = result.get('confidence', 0)
        
        print(f'{i}. "{text}"')
        print(f'   → {sentiment} ({confidence:.1%})')
        
        if sentiment != "Positive":
            print(f'   ⚠️ Issue: Genuinely positive text classified as {sentiment}')
        print()

    return results

def analyze_sarcasm_problem():
    """Analyze why sarcasm is classified as negative"""
    
    print('\n🔍 Analysis: Why Sarcasm → Negative Classification')
    print('=' * 50)
    
    reasons = [
        "1. **Training Data Bias**: Your model was likely trained on data where sarcastic content was labeled as negative",
        "2. **Language Patterns**: Sarcastic text often contains negative words ('great', 'perfect') used ironically",
        "3. **Missing Context**: The model lacks understanding of irony and sarcastic intent",
        "4. **Emoji Misinterpretation**: Sarcastic emojis (🙄😏🙃) may be associated with negative sentiment",
        "5. **Binary Classification**: Your model has only 3 classes (Pos/Neg/Neutral) - no 'Sarcastic' class"
    ]
    
    for reason in reasons:
        print(reason)
    
    print('\n🎯 Is This Actually a Problem?')
    print('=' * 30)
    
    considerations = [
        "✅ **Sarcasm can be negative**: Many sarcastic memes express frustration or criticism",
        "✅ **Intent vs. Reception**: Sarcastic content might be received negatively by some users",
        "⚠️ **Context matters**: Some sarcasm is playful/humorous, not truly negative",
        "⚠️ **Meme culture**: In memes, sarcasm is often used for humor, not genuine negativity"
    ]
    
    for consideration in considerations:
        print(consideration)

if __name__ == "__main__":
    results = test_sarcasm_classification()
    analyze_sarcasm_problem()
    
    print('\n📊 Summary:')
    print('=' * 20)
    negative_count = sum(1 for r in results if r['actual'] == 'Negative')
    total_count = len(results)
    
    print(f'Sarcastic examples classified as Negative: {negative_count}/{total_count} ({negative_count/total_count*100:.1f}%)')
    print('\nThis confirms the issue: sarcastic content is predominantly classified as negative.')
