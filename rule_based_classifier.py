#!/usr/bin/env python3
"""
Rule-Based Sentiment Classifier - Fallback for biased model
This provides better results than the current biased model
"""

import re
import string
from typing import Dict, List

class RuleBasedSentimentClassifier:
    """Simple rule-based sentiment classifier for memes"""
    
    def __init__(self):
        self.positive_keywords = {
            'love', 'amazing', 'awesome', 'great', 'excellent', 'fantastic', 'wonderful',
            'hilarious', 'funny', 'lol', 'haha', 'lmao', 'rofl', 'cute', 'sweet',
            'wholesome', 'beautiful', 'perfect', 'best', 'epic', 'incredible',
            'brilliant', 'clever', 'genius', 'happy', 'joy', 'smile', 'laugh',
            'good', 'nice', 'cool', 'rad', 'dope', 'lit', 'fire', 'blessed'
        }
        
        self.negative_keywords = {
            'hate', 'awful', 'terrible', 'horrible', 'worst', 'bad', 'stupid',
            'dumb', 'annoying', 'frustrating', 'angry', 'mad', 'sad', 'cry',
            'depressed', 'upset', 'disappointed', 'boring', 'lame', 'cringe',
            'gross', 'disgusting', 'pathetic', 'miserable', 'sucks', 'fail',
            'disaster', 'nightmare', 'pain', 'suffering', 'death', 'kill'
        }
        
        self.neutral_keywords = {
            'wednesday', 'thursday', 'friday', 'monday', 'tuesday', 'saturday', 'sunday',
            'today', 'tomorrow', 'yesterday', 'when', 'what', 'where', 'how',
            'just', 'only', 'maybe', 'perhaps', 'sometimes', 'usually',
            'often', 'always', 'never', 'here', 'there', 'now', 'then'
        }
        
        self.positive_emojis = {
            '😂', '🤣', '😄', '😃', '😁', '😊', '😍', '🥰', '😘', '🤗',
            '👍', '👏', '🙌', '💯', '🔥', '⭐', '✨', '❤️', '💕', '💖'
        }
        
        self.negative_emojis = {
            '😭', '😢', '😞', '😔', '😟', '😕', '🙁', '☹️', '😤', '😠',
            '😡', '🤬', '💀', '👎', '😵', '😖', '😣', '😩', '😫', '😰'
        }
        
        self.neutral_emojis = {
            '🤔', '😐', '😑', '🙄', '😒', '😶', '🤨', '🧐', '😯', '😮'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.findall(text)
    
    def count_keywords(self, text: str, keywords: set) -> int:
        """Count occurrences of keywords in text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in keywords)
    
    def count_emojis(self, emojis: List[str], emoji_set: set) -> int:
        """Count emojis from specific sentiment set"""
        return sum(1 for emoji in emojis if emoji in emoji_set)
    
    def analyze_punctuation(self, text: str) -> Dict[str, int]:
        """Analyze punctuation patterns"""
        return {
            'exclamation': text.count('!'),
            'question': text.count('?'),
            'caps_words': len(re.findall(r'\b[A-Z]{2,}\b', text))
        }
    
    def classify_sentiment(self, text: str) -> Dict:
        """Classify sentiment using rules"""
        if not text or not text.strip():
            return {
                'sentiment': 'Neutral',
                'confidence': 0.5,
                'reasoning': 'Empty or whitespace-only text'
            }
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        emojis = self.extract_emojis(text)
        punctuation = self.analyze_punctuation(text)
        
        # Count sentiment indicators
        positive_words = self.count_keywords(processed_text, self.positive_keywords)
        negative_words = self.count_keywords(processed_text, self.negative_keywords)
        neutral_words = self.count_keywords(processed_text, self.neutral_keywords)
        
        positive_emojis = self.count_emojis(emojis, self.positive_emojis)
        negative_emojis = self.count_emojis(emojis, self.negative_emojis)
        neutral_emojis = self.count_emojis(emojis, self.neutral_emojis)
        
        # Calculate scores
        positive_score = positive_words + positive_emojis + punctuation['exclamation'] * 0.1
        negative_score = negative_words + negative_emojis
        neutral_score = neutral_words + neutral_emojis + punctuation['question'] * 0.1
        
        # Adjust for caps (often indicates excitement or anger)
        if punctuation['caps_words'] > 0:
            if positive_score > negative_score:
                positive_score += punctuation['caps_words'] * 0.2
            else:
                negative_score += punctuation['caps_words'] * 0.2
        
        # Determine sentiment
        total_score = positive_score + negative_score + neutral_score
        
        if total_score == 0:
            # No clear indicators, classify as neutral
            sentiment = 'Neutral'
            confidence = 0.6
            reasoning = 'No clear sentiment indicators found'
        else:
            # Calculate relative scores
            pos_ratio = positive_score / total_score
            neg_ratio = negative_score / total_score
            neu_ratio = neutral_score / total_score
            
            if pos_ratio > neg_ratio and pos_ratio > neu_ratio:
                sentiment = 'Positive'
                confidence = min(0.9, 0.5 + pos_ratio)
                reasoning = f'Positive indicators: {positive_words} words, {positive_emojis} emojis'
            elif neg_ratio > neu_ratio:
                sentiment = 'Negative'
                confidence = min(0.9, 0.5 + neg_ratio)
                reasoning = f'Negative indicators: {negative_words} words, {negative_emojis} emojis'
            else:
                sentiment = 'Neutral'
                confidence = min(0.9, 0.5 + neu_ratio)
                reasoning = f'Neutral indicators: {neutral_words} words, {neutral_emojis} emojis'
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'reasoning': reasoning,
            'scores': {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score
            }
        }

def test_rule_based_classifier():
    """Test the rule-based classifier"""
    classifier = RuleBasedSentimentClassifier()
    
    test_cases = [
        "This meme is absolutely hilarious! 😂😂😂",
        "I hate everything about this stupid day",
        "It's Wednesday my dudes",
        "Best meme ever! Love it so much!",
        "Ugh, another boring meeting",
        "This is amazing! 🔥💯",
        "So sad and depressing 😭",
        "What time is it?",
        "LOL this made my day! ❤️",
        "This sucks so much 👎"
    ]
    
    print("🔧 Testing Rule-Based Classifier")
    print("=" * 50)
    
    correct = 0
    total = len(test_cases)
    
    expected = ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 
               'Positive', 'Negative', 'Neutral', 'Positive', 'Negative']
    
    for i, (text, expected_sentiment) in enumerate(zip(test_cases, expected)):
        result = classifier.classify_sentiment(text)
        predicted = result['sentiment']
        confidence = result['confidence']
        reasoning = result['reasoning']
        
        is_correct = predicted == expected_sentiment
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n{status} Test {i+1}: {text}")
        print(f"   Predicted: {predicted} (Expected: {expected_sentiment})")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Reasoning: {reasoning}")
    
    accuracy = correct / total
    print(f"\n📊 Rule-Based Classifier Results:")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    test_rule_based_classifier()
