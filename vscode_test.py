#!/usr/bin/env python3
"""
VS Code Interactive Test Script for Multimodal Meme Sentiment Analysis
Run this in VS Code to test your model interactively
"""

import os
import json
import sys
from datetime import datetime
from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer

def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "="*60)
    if title:
        print(f"🎯 {title}")
        print("="*60)

def print_result(result):
    """Pretty print analysis results"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Check if it's image/video result or text result
    if "final_sentiment" in result:
        # Image/Video results
        sentiment = result["final_sentiment"]
        confidence = result.get("final_confidence", 0)
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "❓")
        
        print(f"\n🎯 Final Result: {emoji} {sentiment} ({confidence:.1%} confidence)")
        
        breakdown = result.get("analysis_breakdown", {})
        
        if breakdown.get("extracted_text"):
            text = breakdown["extracted_text"][:100].replace('\n', ' ')
            print(f"📝 Extracted Text: \"{text}...\"")
        
        if breakdown.get("image_caption"):
            print(f"🖼️  Image Caption: \"{breakdown['image_caption']}\"")
        
        print(f"\n🔍 Analysis Breakdown:")
        if breakdown.get("text_sentiment") and "sentiment" in breakdown["text_sentiment"]:
            ts = breakdown["text_sentiment"]
            print(f"   📝 Text: {ts['sentiment']} ({ts['confidence']:.1%})")
        
        if breakdown.get("visual_sentiment") and "sentiment" in breakdown["visual_sentiment"]:
            vs = breakdown["visual_sentiment"]
            print(f"   👁️  Visual: {vs['sentiment']} ({vs['confidence']:.1%})")
        
        if breakdown.get("caption_sentiment") and "sentiment" in breakdown["caption_sentiment"]:
            cs = breakdown["caption_sentiment"]
            print(f"   🖼️  Caption: {cs['sentiment']} ({cs['confidence']:.1%})")
        
        if "meme_categories" in result:
            print(f"🏷️  Categories: {', '.join(result['meme_categories'])}")
    
    else:
        # Text results
        sentiment = result.get("sentiment", "Unknown")
        confidence = result.get("confidence", 0)
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "❓")
        
        print(f"\n🎯 Result: {emoji} {sentiment} ({confidence:.1%} confidence)")
        
        if "all_scores" in result:
            print(f"\n📊 All Scores:")
            scores = result["all_scores"]
            for label, score in scores.items():
                bar = "█" * int(score * 20)  # Simple bar chart
                print(f"   {label:8}: {score:.1%} {bar}")

def test_text_samples():
    """Test various text samples"""
    print_separator("Text Analysis Tests")
    
    test_texts = [
        "This meme is absolutely hilarious! 😂🤣",
        "I hate this stupid meme, it's so annoying",
        "This is just a regular meme about cats",
        "LMAO this made my day! So funny!",
        "This meme is cringe and not funny at all",
        "Meh, it's an okay meme I guess",
        "Best meme ever! 💯🔥",
        "This makes me sad 😢",
        "Pretty wholesome content ❤️"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: \"{text}\"")
        print("-" * 50)
        try:
            result = analyzer.analyze_meme(text, "text")
            print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")

def test_image_samples():
    """Test available image samples"""
    print_separator("Image Analysis Tests")
    
    if not os.path.exists("meme_dataset"):
        print("❌ No meme_dataset folder found")
        return
    
    # Get all image files
    image_files = []
    for file in os.listdir("meme_dataset"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print("❌ No image files found in meme_dataset")
        return
    
    # Test up to 5 images
    for i, image_file in enumerate(image_files[:5], 1):
        image_path = os.path.join("meme_dataset", image_file)
        print(f"\n🖼️  Test {i}: {image_file}")
        print("-" * 50)
        try:
            result = analyzer.analyze_meme(image_path, "image")
            print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")

def test_auto_detection():
    """Test auto-detection feature"""
    print_separator("Auto-Detection Tests")
    
    test_inputs = [
        "This is a text meme!",
        "meme_dataset/positive_1.jpg" if os.path.exists("meme_dataset/positive_1.jpg") else None,
    ]
    
    for i, input_data in enumerate(test_inputs, 1):
        if input_data is None:
            continue
            
        print(f"\n🔄 Auto-Detection Test {i}: {input_data}")
        print("-" * 50)
        try:
            result = analyzer.analyze_meme(input_data)  # No type specified
            detected_type = result.get("input_type", "unknown")
            print(f"✅ Detected Type: {detected_type}")
            print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")

def interactive_test():
    """Interactive testing mode"""
    print_separator("Interactive Testing Mode")
    print("Enter 'quit' to exit")
    
    while True:
        print("\n" + "-"*40)
        user_input = input("Enter text to analyze (or file path): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        print(f"\n🔄 Analyzing: {user_input}")
        try:
            result = analyzer.analyze_meme(user_input)
            print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")

def performance_test():
    """Test performance with timing"""
    print_separator("Performance Testing")
    
    import time
    
    test_cases = [
        ("Text", "This meme is funny!", "text"),
        ("Image", "meme_dataset/positive_1.jpg", "image") if os.path.exists("meme_dataset/positive_1.jpg") else None
    ]
    
    for test_case in test_cases:
        if test_case is None:
            continue
            
        test_type, input_data, input_type = test_case
        print(f"\n⚡ {test_type} Performance Test")
        print("-" * 30)
        
        start_time = time.time()
        try:
            result = analyzer.analyze_meme(input_data, input_type)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"✅ Processing Time: {processing_time:.2f} seconds")
            
            if "final_sentiment" in result:
                print(f"✅ Result: {result['final_sentiment']} ({result.get('final_confidence', 0):.1%})")
            else:
                print(f"✅ Result: {result.get('sentiment', 'Unknown')} ({result.get('confidence', 0):.1%})")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def save_test_results():
    """Save comprehensive test results"""
    print_separator("Saving Test Results")
    
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "model_path": analyzer.text_model_path,
            "models_loaded": "CLIP, BLIP, DistilBERT, OpenCV"
        },
        "test_results": []
    }
    
    # Test a few samples and save results
    samples = [
        ("This meme is hilarious!", "text"),
        ("This meme sucks", "text"),
        ("meme_dataset/positive_1.jpg", "image") if os.path.exists("meme_dataset/positive_1.jpg") else None
    ]
    
    for sample in samples:
        if sample is None:
            continue
            
        input_data, input_type = sample
        try:
            result = analyzer.analyze_meme(input_data, input_type)
            test_data["test_results"].append({
                "input": input_data,
                "type": input_type,
                "result": result
            })
        except Exception as e:
            test_data["test_results"].append({
                "input": input_data,
                "type": input_type,
                "error": str(e)
            })
    
    # Save to file
    with open("vscode_test_results.json", "w") as f:
        json.dump(test_data, f, indent=2, default=str)
    
    print("📄 Results saved to: vscode_test_results.json")

def main():
    """Main testing function for VS Code"""
    print("🎯 VS Code Multimodal Sentiment Analysis Testing")
    print("="*60)
    print("This script will test your multimodal sentiment analysis system")
    print("Run each section to verify everything is working correctly")
    
    global analyzer
    
    print("\n🔄 Initializing multimodal analyzer...")
    try:
        analyzer = AdvancedMultimodalAnalyzer()
        print("✅ Analyzer initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check if production_model_v2 folder exists")
        print("3. Verify Tesseract OCR is installed")
        return
    
    # Run different test sections
    print("\n" + "="*60)
    print("Choose what to test:")
    print("1. Run all tests")
    print("2. Text analysis only") 
    print("3. Image analysis only")
    print("4. Auto-detection tests")
    print("5. Performance tests")
    print("6. Interactive mode")
    print("7. Save test results")
    
    choice = input("\nEnter your choice (1-7) or press Enter for all tests: ").strip()
    
    if choice == "2":
        test_text_samples()
    elif choice == "3":
        test_image_samples()
    elif choice == "4":
        test_auto_detection()
    elif choice == "5":
        performance_test()
    elif choice == "6":
        interactive_test()
    elif choice == "7":
        save_test_results()
    else:
        # Run all tests
        test_text_samples()
        test_image_samples()
        test_auto_detection()
        performance_test()
        save_test_results()
    
    print_separator("Testing Complete")
    print("🎉 All tests finished!")
    print("Your multimodal sentiment analysis system is working correctly.")

if __name__ == "__main__":
    main()
