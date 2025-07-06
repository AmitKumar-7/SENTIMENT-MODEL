#!/usr/bin/env python3
"""
Quick test of multimodal image analysis
"""

from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer
import json

def test_multimodal_image():
    """Test multimodal analysis on a specific image"""
    
    print('🎯 Testing Advanced Multimodal Analysis')
    print('=' * 50)

    # Initialize analyzer
    analyzer = AdvancedMultimodalAnalyzer()

    # Test with different image types
    test_images = [
        'meme_dataset/positive_1.jpg',
        'meme_dataset/negative_2.jpg'
    ]
    
    for img_path in test_images:
        print(f'\n📸 Testing Image: {img_path}')
        print('-' * 40)
        
        try:
            result = analyzer.analyze_meme(img_path, 'image')
            
            if 'error' in result:
                print(f'❌ Error: {result["error"]}')
                continue
                
            print(f'✅ Final Sentiment: {result["final_sentiment"]}')
            print(f'✅ Final Confidence: {result["final_confidence"]:.3f}')
            
            breakdown = result['analysis_breakdown']
            
            if breakdown.get('extracted_text'):
                text = breakdown['extracted_text'][:80].replace('\n', ' ')
                print(f'📝 Extracted Text: "{text}..."')
            
            if breakdown.get('image_caption'):
                print(f'🖼️  Image Caption: "{breakdown["image_caption"]}"')
            
            if breakdown.get('text_sentiment') and 'sentiment' in breakdown['text_sentiment']:
                ts = breakdown['text_sentiment']
                print(f'📝 Text Sentiment: {ts["sentiment"]} ({ts["confidence"]:.3f})')
            
            if breakdown.get('visual_sentiment') and 'sentiment' in breakdown['visual_sentiment']:
                vs = breakdown['visual_sentiment']
                print(f'👁️  Visual Sentiment: {vs["sentiment"]} ({vs["confidence"]:.3f})')
                
            if breakdown.get('face_emotions'):
                faces = breakdown['face_emotions']['faces_detected']
                print(f'👤 Faces Detected: {faces}')
                
        except Exception as e:
            print(f'❌ Error analyzing {img_path}: {str(e)}')

    # Test text analysis for comparison
    print(f'\n📝 Testing Text Analysis for Comparison:')
    print('-' * 40)
    
    test_texts = [
        "This meme is absolutely hilarious! 😂",
        "I hate this stupid meme",
        "This is a wholesome and funny meme"
    ]
    
    for text in test_texts:
        try:
            result = analyzer.analyze_meme(text, 'text')
            sentiment = result.get('sentiment', 'Unknown')
            confidence = result.get('confidence', 0)
            print(f'"{text[:30]}..." → {sentiment} ({confidence:.3f})')
        except Exception as e:
            print(f'❌ Error: {str(e)}')

if __name__ == "__main__":
    test_multimodal_image()
