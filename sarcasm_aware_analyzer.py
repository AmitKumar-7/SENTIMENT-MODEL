#!/usr/bin/env python3
"""
Sarcasm-Aware Sentiment Analyzer
Enhanced version that can detect and properly classify sarcastic content
"""

import re
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SarcasmAwareAnalyzer(AdvancedMultimodalAnalyzer):
    """
    Enhanced analyzer with sarcasm detection capabilities
    """
    
    def __init__(self, text_model_path: str = None):
        super().__init__(text_model_path)
        
        # Sarcasm indicators
        self.sarcasm_patterns = {
            'emojis': ['🙄', '😏', '🙃', '😒', '🤨', '😑', '😐', '🤔', '🤡'],
            'phrases': [
                r'\boh (great|wonderful|fantastic|perfect|amazing)\b',
                r'\bwow[,\s]+(such|so|very)',
                r'\byeah[,\s]+(right|sure|because)',
                r'\boh sure\b',
                r'\bof course\b',
                r'\bobviously\b',
                r'\bthanks[,\s]+(captain obvious|genius)',
                r'\bas if\b',
                r'\byeah[,\s]+that(\'s| is) gonna work',
                r'\breal(ly)? (original|creative|helpful)\b',
                r'\bjust what I needed\b',
                r'\bperfect timing\b'
            ],
            'punctuation_patterns': [
                r'\.{3,}',  # Multiple dots
                r'!{2,}',   # Multiple exclamation marks
                r'\?{2,}',  # Multiple question marks
            ],
            'caps_patterns': [
                r'\b[A-Z]{3,}\b',  # All caps words
            ]
        }
        
        # Context clues for meme sarcasm
        self.meme_sarcasm_indicators = [
            'another', 'monday', 'work', 'boss', 'assignment', 'deadline',
            'meeting', 'traffic', 'weather', 'politicians', 'news',
            'social media', 'internet', 'technology', 'generation'
        ]
        
    def detect_sarcasm_score(self, text: str) -> float:
        """
        Calculate sarcasm probability score based on linguistic patterns
        
        Returns:
            float: Sarcasm score between 0 and 1
        """
        sarcasm_score = 0.0
        text_lower = text.lower()
        
        # Check for sarcastic emojis
        emoji_count = sum(1 for emoji in self.sarcasm_patterns['emojis'] if emoji in text)
        sarcasm_score += emoji_count * 0.3
        
        # Check for sarcastic phrases
        phrase_matches = 0
        for pattern in self.sarcasm_patterns['phrases']:
            if re.search(pattern, text_lower):
                phrase_matches += 1
        sarcasm_score += phrase_matches * 0.4
        
        # Check punctuation patterns
        punct_matches = 0
        for pattern in self.sarcasm_patterns['punctuation_patterns']:
            if re.search(pattern, text):
                punct_matches += 1
        sarcasm_score += punct_matches * 0.2
        
        # Check for caps (mild indicator)
        caps_matches = len(re.findall(r'\b[A-Z]{3,}\b', text))
        sarcasm_score += caps_matches * 0.1
        
        # Check for meme context indicators
        meme_context = sum(1 for indicator in self.meme_sarcasm_indicators 
                          if indicator in text_lower)
        if meme_context > 0:
            sarcasm_score += 0.2
        
        # Normalize score to 0-1 range
        return min(sarcasm_score, 1.0)
    
    def classify_sarcasm_type(self, text: str, sentiment: str) -> str:
        """
        Classify the type of sarcasm based on content and context
        """
        text_lower = text.lower()
        
        # Playful sarcasm (more humorous)
        if any(word in text_lower for word in ['lol', 'lmao', 'haha', 'funny', 'joke']):
            return "playful_sarcasm"
        
        # Frustrated sarcasm (genuinely negative)
        if any(word in text_lower for word in ['work', 'boss', 'monday', 'deadline', 'traffic']):
            return "frustrated_sarcasm"
        
        # Critical sarcasm (mocking/critical)
        if any(word in text_lower for word in ['original', 'creative', 'genius', 'brilliant']):
            return "critical_sarcasm"
        
        # Mild sarcasm (neutral-ish)
        return "mild_sarcasm"
    
    def adjust_sentiment_for_sarcasm(self, text: str, original_result: Dict) -> Dict:
        """
        Adjust sentiment classification when sarcasm is detected
        """
        sarcasm_score = self.detect_sarcasm_score(text)
        
        if sarcasm_score < 0.3:
            # Low sarcasm probability, keep original result
            return original_result
        
        # High sarcasm detected, adjust the result
        adjusted_result = original_result.copy()
        original_sentiment = original_result.get('sentiment', 'Unknown')
        original_confidence = original_result.get('confidence', 0)
        
        # Classify sarcasm type
        sarcasm_type = self.classify_sarcasm_type(text, original_sentiment)
        
        # Adjust based on sarcasm type
        if sarcasm_type == "playful_sarcasm":
            # Playful sarcasm in memes is often neutral to positive
            adjusted_result['sentiment'] = 'Neutral'
            adjusted_result['confidence'] = 0.6
            adjusted_result['sarcasm_detected'] = True
            adjusted_result['sarcasm_type'] = 'Playful'
            adjusted_result['sarcasm_score'] = sarcasm_score
            
        elif sarcasm_type == "frustrated_sarcasm":
            # Keep as negative but acknowledge it's sarcastic
            adjusted_result['sentiment'] = 'Negative'
            adjusted_result['confidence'] = max(0.7, original_confidence)
            adjusted_result['sarcasm_detected'] = True
            adjusted_result['sarcasm_type'] = 'Frustrated'
            adjusted_result['sarcasm_score'] = sarcasm_score
            
        elif sarcasm_type == "critical_sarcasm":
            # Critical sarcasm is negative but not as harsh
            adjusted_result['sentiment'] = 'Negative'
            adjusted_result['confidence'] = 0.6
            adjusted_result['sarcasm_detected'] = True
            adjusted_result['sarcasm_type'] = 'Critical'
            adjusted_result['sarcasm_score'] = sarcasm_score
            
        else:  # mild_sarcasm
            # Mild sarcasm -> neutral
            adjusted_result['sentiment'] = 'Neutral'
            adjusted_result['confidence'] = 0.5
            adjusted_result['sarcasm_detected'] = True
            adjusted_result['sarcasm_type'] = 'Mild'
            adjusted_result['sarcasm_score'] = sarcasm_score
        
        return adjusted_result
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Enhanced text sentiment analysis with sarcasm detection
        """
        # Get original sentiment analysis
        original_result = super().analyze_text_sentiment(text)
        
        if "error" in original_result:
            return original_result
        
        # Apply sarcasm detection and adjustment
        adjusted_result = self.adjust_sentiment_for_sarcasm(text, original_result)
        
        # Add original classification for comparison
        adjusted_result['original_sentiment'] = original_result.get('sentiment')
        adjusted_result['original_confidence'] = original_result.get('confidence')
        
        return adjusted_result
    
    def analyze_meme(self, input_data: str, input_type: str = "auto") -> Dict:
        """
        Enhanced meme analysis with sarcasm awareness
        """
        result = super().analyze_meme(input_data, input_type)
        
        # If it's text analysis, sarcasm is already handled in analyze_text_sentiment
        if input_type == "text" or (input_type == "auto" and not os.path.isfile(input_data)):
            return result
        
        # For image/video analysis, check if extracted text has sarcasm
        if "analysis_breakdown" in result:
            breakdown = result["analysis_breakdown"]
            
            # Check extracted text for sarcasm
            extracted_text = breakdown.get("extracted_text", "")
            if extracted_text:
                sarcasm_score = self.detect_sarcasm_score(extracted_text)
                
                if sarcasm_score >= 0.3:
                    # Sarcasm detected in image text
                    result["sarcasm_detected_in_image"] = True
                    result["image_sarcasm_score"] = sarcasm_score
                    
                    # Adjust final sentiment if confidence is low
                    final_confidence = result.get("final_confidence", 0)
                    if final_confidence < 0.6:
                        sarcasm_type = self.classify_sarcasm_type(extracted_text, result["final_sentiment"])
                        
                        if sarcasm_type in ["playful_sarcasm", "mild_sarcasm"]:
                            result["final_sentiment"] = "Neutral"
                            result["final_confidence"] = 0.6
                            result["adjustment_reason"] = f"Sarcasm detected: {sarcasm_type}"
        
        return result

def test_sarcasm_improvements():
    """Test the improved sarcasm handling"""
    
    print("🎯 Testing Sarcasm-Aware Analyzer")
    print("=" * 50)
    
    # Initialize both analyzers for comparison
    original_analyzer = AdvancedMultimodalAnalyzer()
    sarcasm_analyzer = SarcasmAwareAnalyzer()
    
    test_cases = [
        "Oh great, another Monday meme 🙄",
        "Wow, such original content 😏",
        "Yeah, because that always works out perfectly",
        "Perfect timing, just what I needed today",
        "Thanks for the reminder, as if I could forget 🙃",
        "This meme is genuinely hilarious!",  # Should stay positive
        "I love this wholesome content ❤️",   # Should stay positive
        "This is terrible and not funny",     # Should stay negative
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. \"{text}\"")
        print("-" * 40)
        
        # Original analysis
        original_result = original_analyzer.analyze_text_sentiment(text)
        original_sentiment = original_result.get('sentiment', 'Unknown')
        original_confidence = original_result.get('confidence', 0)
        
        # Sarcasm-aware analysis
        sarcasm_result = sarcasm_analyzer.analyze_text_sentiment(text)
        new_sentiment = sarcasm_result.get('sentiment', 'Unknown')
        new_confidence = sarcasm_result.get('confidence', 0)
        
        print(f"Original:  {original_sentiment} ({original_confidence:.1%})")
        print(f"Enhanced:  {new_sentiment} ({new_confidence:.1%})")
        
        # Show sarcasm detection details
        if sarcasm_result.get('sarcasm_detected'):
            sarcasm_type = sarcasm_result.get('sarcasm_type', 'Unknown')
            sarcasm_score = sarcasm_result.get('sarcasm_score', 0)
            print(f"Sarcasm:   Detected - {sarcasm_type} (score: {sarcasm_score:.2f})")
            
            if original_sentiment != new_sentiment:
                print(f"✅ Improved: {original_sentiment} → {new_sentiment}")
            else:
                print(f"➡️ Confirmed: {new_sentiment} (sarcasm acknowledged)")
        else:
            print(f"Sarcasm:   Not detected")
            if original_sentiment != new_sentiment:
                print(f"⚠️ Changed without sarcasm: {original_sentiment} → {new_sentiment}")

if __name__ == "__main__":
    test_sarcasm_improvements()
