#!/usr/bin/env python3
"""
Balanced Sentiment Analyzer
Enhanced version that fixes the negative bias with rule-based positive detection
"""

import re
import os
import torch
import numpy as np
from typing import Dict, List
from sarcasm_aware_analyzer import SarcasmAwareAnalyzer
import logging

logger = logging.getLogger(__name__)

class BalancedSentimentAnalyzer(SarcasmAwareAnalyzer):
    """
    Enhanced analyzer that fixes negative bias with rule-based positive detection
    """
    
    def __init__(self, text_model_path: str = None):
        super().__init__(text_model_path)
        
        # Positive sentiment indicators
        self.positive_indicators = {
            'strong_positive_words': [
                'love', 'amazing', 'awesome', 'fantastic', 'brilliant', 'excellent',
                'wonderful', 'perfect', 'beautiful', 'incredible', 'outstanding',
                'hilarious', 'funny', 'clever', 'creative', 'genius', 'adorable',
                'heartwarming', 'wholesome', 'delightful', 'charming', 'sweet'
            ],
            'positive_phrases': [
                r'\bmade my day\b',
                r'\bso (funny|good|great|amazing)\b',
                r'\blove (this|it)\b',
                r'\bthis is (great|amazing|awesome|funny|hilarious)\b',
                r'\bactually (love|like|enjoy)\b',
                r'\bgenuinely (funny|good|great|amazing)\b',
                r'\breally (love|like|enjoy|funny|good)\b',
                r'\bthanks for (sharing|this)\b',
                r'\bbest (meme|content) ever\b',
                r'\bcan\'t stop laughing\b',
                r'\bcracked me up\b'
            ],
            'positive_emojis': [
                '😂', '🤣', '😊', '😍', '🥰', '😘', '❤️', '💕', '💖', '💯',
                '🔥', '👌', '👍', '🙌', '🎉', '✨', '⭐', '🌟', '💫', '🤩'
            ],
            'enthusiasm_indicators': [
                r'!{2,}',  # Multiple exclamation marks
                r'\b(haha|lol|lmao|rofl)\b',
                r'\b(yay|woohoo|awesome)\b',
                r'\bso (much|many)\b'
            ]
        }
        
        # Neutral indicators
        self.neutral_indicators = {
            'factual_phrases': [
                r'\bthis is (a|an) (meme|picture|video|content) about\b',
                r'\bstandard (meme|format|content)\b',
                r'\b(this|that) (meme|content|picture) (exists|shows|depicts)\b',
                r'\bregular (meme|content|format)\b',
                r'\btypical (meme|content)\b',
                r'\bjust (a|an) (meme|picture|video)\b'
            ],
            'neutral_words': [
                'standard', 'regular', 'typical', 'normal', 'basic', 'simple',
                'exists', 'shows', 'depicts', 'about', 'contains', 'format'
            ]
        }
        
    def detect_positive_sentiment(self, text: str) -> Dict:
        """
        Rule-based positive sentiment detection to fix model bias
        """
        text_lower = text.lower()
        positive_score = 0.0
        evidence = []
        
        # Check for strong positive words
        positive_word_count = 0
        for word in self.positive_indicators['strong_positive_words']:
            if word in text_lower:
                positive_word_count += 1
                evidence.append(f"positive_word: {word}")
        positive_score += positive_word_count * 0.3
        
        # Check for positive phrases
        positive_phrase_count = 0
        for pattern in self.positive_indicators['positive_phrases']:
            if re.search(pattern, text_lower):
                positive_phrase_count += 1
                evidence.append(f"positive_phrase: {pattern}")
        positive_score += positive_phrase_count * 0.4
        
        # Check for positive emojis
        positive_emoji_count = 0
        for emoji in self.positive_indicators['positive_emojis']:
            if emoji in text:
                positive_emoji_count += 1
                evidence.append(f"positive_emoji: {emoji}")
        positive_score += positive_emoji_count * 0.3
        
        # Check for enthusiasm indicators
        enthusiasm_count = 0
        for pattern in self.positive_indicators['enthusiasm_indicators']:
            if re.search(pattern, text_lower):
                enthusiasm_count += 1
                evidence.append(f"enthusiasm: {pattern}")
        positive_score += enthusiasm_count * 0.2
        
        return {
            'positive_score': min(positive_score, 1.0),
            'evidence': evidence,
            'detected': positive_score >= 0.4
        }
    
    def detect_neutral_sentiment(self, text: str) -> Dict:
        """
        Rule-based neutral sentiment detection
        """
        text_lower = text.lower()
        neutral_score = 0.0
        evidence = []
        
        # Check for factual phrases
        factual_count = 0
        for pattern in self.neutral_indicators['factual_phrases']:
            if re.search(pattern, text_lower):
                factual_count += 1
                evidence.append(f"factual_phrase: {pattern}")
        neutral_score += factual_count * 0.5
        
        # Check for neutral words
        neutral_word_count = 0
        for word in self.neutral_indicators['neutral_words']:
            if word in text_lower:
                neutral_word_count += 1
                evidence.append(f"neutral_word: {word}")
        neutral_score += neutral_word_count * 0.2
        
        # Short, simple statements tend to be neutral
        if len(text.split()) <= 6 and not any(char in text for char in ['!', '?', '😂', '❤️']):
            neutral_score += 0.3
            evidence.append("short_simple_statement")
        
        return {
            'neutral_score': min(neutral_score, 1.0),
            'evidence': evidence,
            'detected': neutral_score >= 0.4
        }
    
    def apply_bias_correction(self, text: str, original_result: Dict) -> Dict:
        """
        Apply bias correction using rule-based detection
        """
        # Get rule-based detections
        positive_detection = self.detect_positive_sentiment(text)
        neutral_detection = self.detect_neutral_sentiment(text)
        
        # Get original results
        original_sentiment = original_result.get('sentiment', 'Unknown')
        original_confidence = original_result.get('confidence', 0)
        
        # Start with original result
        corrected_result = original_result.copy()
        
        # Apply corrections based on rule-based detection
        if positive_detection['detected'] and original_sentiment.lower() != 'positive':
            # Strong positive indicators found but model said negative/neutral
            corrected_result['sentiment'] = 'Positive'
            corrected_result['confidence'] = 0.7 + (positive_detection['positive_score'] * 0.2)
            corrected_result['bias_correction'] = {
                'applied': True,
                'reason': 'positive_indicators_detected',
                'original_sentiment': original_sentiment,
                'positive_evidence': positive_detection['evidence']
            }
            
        elif neutral_detection['detected'] and original_sentiment.lower() == 'negative' and original_confidence < 0.5:
            # Neutral indicators + low confidence negative = likely neutral
            corrected_result['sentiment'] = 'Neutral'
            corrected_result['confidence'] = 0.6
            corrected_result['bias_correction'] = {
                'applied': True,
                'reason': 'neutral_indicators_detected',
                'original_sentiment': original_sentiment,
                'neutral_evidence': neutral_detection['evidence']
            }
        
        else:
            # No correction needed
            corrected_result['bias_correction'] = {
                'applied': False,
                'reason': 'no_strong_indicators_found'
            }
        
        return corrected_result
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Enhanced text sentiment analysis with bias correction
        """
        # Get sarcasm-aware result first
        sarcasm_result = super().analyze_text_sentiment(text)
        
        if "error" in sarcasm_result:
            return sarcasm_result
        
        # Apply bias correction
        corrected_result = self.apply_bias_correction(text, sarcasm_result)
        
        # Add original classification for comparison
        corrected_result['pre_correction_sentiment'] = sarcasm_result.get('sentiment')
        corrected_result['pre_correction_confidence'] = sarcasm_result.get('confidence')
        
        return corrected_result

def test_bias_correction():
    """Test the bias correction functionality"""
    
    print("🎯 Testing Bias Correction")
    print("=" * 50)
    
    # Initialize analyzers
    original_analyzer = SarcasmAwareAnalyzer()
    balanced_analyzer = BalancedSentimentAnalyzer()
    
    test_cases = [
        # Should be positive
        {"text": "This meme is absolutely hilarious and made my day!", "expected": "positive"},
        {"text": "I love this wholesome content ❤️", "expected": "positive"},
        {"text": "This is genuinely funny and clever", "expected": "positive"},
        {"text": "Such a heartwarming meme!", "expected": "positive"},
        {"text": "LMAO this is so funny! 😂", "expected": "positive"},
        {"text": "Best meme ever! Thanks for sharing", "expected": "positive"},
        
        # Should be neutral
        {"text": "This is a meme about cats", "expected": "neutral"},
        {"text": "Standard meme format", "expected": "neutral"},
        {"text": "This meme exists", "expected": "neutral"},
        {"text": "Regular content", "expected": "neutral"},
        
        # Should stay negative
        {"text": "This meme is terrible and not funny", "expected": "negative"},
        {"text": "I hate this content", "expected": "negative"},
        
        # Sarcastic (should be handled by sarcasm detection)
        {"text": "Oh great, another Monday meme 🙄", "expected": "negative"},
        {"text": "Perfect timing, just what I needed", "expected": "neutral"},
    ]
    
    correct_original = 0
    correct_balanced = 0
    
    for i, test in enumerate(test_cases, 1):
        text = test["text"]
        expected = test["expected"]
        
        # Original analysis
        orig_result = original_analyzer.analyze_text_sentiment(text)
        orig_sentiment = orig_result.get('sentiment', 'Unknown').lower()
        
        # Balanced analysis
        balanced_result = balanced_analyzer.analyze_text_sentiment(text)
        balanced_sentiment = balanced_result.get('sentiment', 'Unknown').lower()
        
        # Check accuracy
        orig_correct = orig_sentiment == expected
        balanced_correct = balanced_sentiment == expected
        
        if orig_correct:
            correct_original += 1
        if balanced_correct:
            correct_balanced += 1
        
        print(f"{i:2d}. \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"    Expected:  {expected.title()}")
        print(f"    Original:  {orig_sentiment.title()} {'✅' if orig_correct else '❌'}")
        print(f"    Balanced:  {balanced_sentiment.title()} {'✅' if balanced_correct else '❌'}")
        
        # Show correction details
        correction = balanced_result.get('bias_correction', {})
        if correction.get('applied'):
            reason = correction.get('reason', 'unknown')
            print(f"    Correction: Applied - {reason}")
            if 'positive_evidence' in correction:
                evidence = correction['positive_evidence'][:2]  # Show first 2 pieces of evidence
                print(f"    Evidence:   {', '.join(evidence)}")
        
        print()
    
    # Summary
    original_accuracy = correct_original / len(test_cases)
    balanced_accuracy = correct_balanced / len(test_cases)
    improvement = balanced_accuracy - original_accuracy
    
    print(f"📊 Results Summary:")
    print(f"   Original Accuracy:  {original_accuracy:.1%}")
    print(f"   Balanced Accuracy:  {balanced_accuracy:.1%}")
    print(f"   Improvement:        {improvement*100:+.1f} percentage points")

if __name__ == "__main__":
    test_bias_correction()
