#!/usr/bin/env python3
"""
🎯 Single Meme Image Analysis
Test a specific meme image with comprehensive multimodal analysis
"""

import sys
import os
from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer
import json

def analyze_single_meme(image_path):
    """Analyze a single meme image comprehensively"""
    
    print('🎯 COMPREHENSIVE MEME ANALYSIS')
    print('=' * 60)
    
    # Get just the filename
    filename = os.path.basename(image_path)
    print(f'📁 Image: {filename}')
    print(f'📍 Path: {image_path}')
    print()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f'❌ Error: File not found: {image_path}')
        return
    
    try:
        # Initialize analyzer
        print('🔄 Initializing multimodal analyzer...')
        analyzer = AdvancedMultimodalAnalyzer()
        print('✅ Analyzer ready!')
        print()
        
        # Run analysis
        print('🔍 Analyzing meme...')
        result = analyzer.analyze_meme(image_path, 'image')
        
        print()
        print('📊 ANALYSIS RESULTS:')
        print('-' * 40)
        print(f'🎯 Final Sentiment: {result["final_sentiment"]}')
        print(f'📈 Final Confidence: {result["final_confidence"]:.1%}')
        print(f'🏷️ Meme Categories: {result["meme_categories"]}')
        print()
        
        breakdown = result['analysis_breakdown']
        
        print('📝 TEXT EXTRACTION:')
        print('-' * 40)
        extracted_text = breakdown.get("extracted_text", "")
        if extracted_text:
            print(f'📄 Extracted Text: "{extracted_text[:200]}..."')
        else:
            print('📄 Extracted Text: [No text detected]')
        print()
        
        print('🖼️ IMAGE ANALYSIS:')
        print('-' * 40)
        caption = breakdown.get("image_caption", "")
        if caption:
            print(f'📷 Generated Caption: "{caption[:150]}..."')
        else:
            print('📷 Generated Caption: [No caption generated]')
        print()
        
        print('🧠 SENTIMENT BREAKDOWN:')
        print('-' * 40)
        text_sent = breakdown['text_sentiment']
        visual_sent = breakdown['visual_sentiment']
        caption_sent = breakdown['caption_sentiment']
        
        print(f'📊 Text Sentiment: {text_sent["sentiment"]} ({text_sent["confidence"]:.1%})')
        print(f'   Scores: Neg:{text_sent["all_scores"]["Negative"]:.3f}, Neu:{text_sent["all_scores"]["Neutral"]:.3f}, Pos:{text_sent["all_scores"]["Positive"]:.3f}')
        print()
        print(f'👁️ Visual Sentiment: {visual_sent["sentiment"]} ({visual_sent["confidence"]:.1%})')
        print(f'   Scores: Neg:{visual_sent["all_scores"]["Negative"]:.3f}, Neu:{visual_sent["all_scores"]["Neutral"]:.3f}, Pos:{visual_sent["all_scores"]["Positive"]:.3f}')
        print()
        print(f'📷 Caption Sentiment: {caption_sent["sentiment"]} ({caption_sent["confidence"]:.1%})')
        print(f'   Scores: Neg:{caption_sent["all_scores"]["Negative"]:.3f}, Neu:{caption_sent["all_scores"]["Neutral"]:.3f}, Pos:{caption_sent["all_scores"]["Positive"]:.3f}')
        print()
        
        print('😊 FACE & EMOTION ANALYSIS:')
        print('-' * 40)
        face_data = breakdown['face_emotions']
        print(f'👤 Faces Detected: {face_data["faces_detected"]}')
        print(f'🎭 Emotions: {face_data["emotions"] if face_data["emotions"] else "[None detected]"}')
        
        print()
        print('✅ COMPREHENSIVE ANALYSIS COMPLETE!')
        
        # Save results
        output_file = f'analysis_result_{filename}.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'💾 Results saved to: {output_file}')
        
    except Exception as e:
        print(f'❌ Error during analysis: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test image path
    image_path = "C:/Users/KIIT/Downloads/download (2).jpg"
    analyze_single_meme(image_path)
