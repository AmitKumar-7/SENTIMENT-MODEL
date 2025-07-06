#!/usr/bin/env python3
"""
Test script for multimodal meme sentiment analysis system
"""

import os
import json
from datetime import datetime
from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer

def test_multimodal_system():
    """Test the multimodal sentiment analysis system"""
    
    print("🎯 Testing Advanced Multimodal Meme Sentiment Analysis")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        print("🔄 Initializing multimodal analyzer...")
        analyzer = AdvancedMultimodalAnalyzer()
        print("✅ Analyzer initialized successfully!")
        
        # Test cases
        test_cases = [
            {
                "type": "text",
                "input": "This meme is absolutely hilarious! 😂",
                "description": "Positive text with emoji"
            },
            {
                "type": "text", 
                "input": "This meme is so stupid and annoying",
                "description": "Negative text sentiment"
            },
            {
                "type": "text",
                "input": "This is just a regular meme about cats",
                "description": "Neutral text content"
            }
        ]
        
        # Add image tests if sample images exist
        if os.path.exists("meme_dataset"):
            sample_images = []
            for file in os.listdir("meme_dataset"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join("meme_dataset", file))
                    if len(sample_images) >= 3:  # Limit to 3 for testing
                        break
            
            for img_path in sample_images:
                test_cases.append({
                    "type": "image",
                    "input": img_path,
                    "description": f"Image analysis: {os.path.basename(img_path)}"
                })
        
        # Run tests
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['description']}")
            print("-" * 50)
            
            try:
                # Analyze the meme
                result = analyzer.analyze_meme(test_case["input"], test_case["type"])
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                # Display results based on type
                if test_case["type"] == "text":
                    print(f"Input: '{test_case['input'][:60]}...'")
                    print(f"Sentiment: {result.get('sentiment', 'N/A')}")
                    print(f"Confidence: {result.get('confidence', 0):.3f}")
                    if "all_scores" in result:
                        scores = result["all_scores"]
                        print(f"All Scores: Neg:{scores.get('Negative', 0):.3f}, "
                              f"Neu:{scores.get('Neutral', 0):.3f}, "
                              f"Pos:{scores.get('Positive', 0):.3f}")
                
                elif test_case["type"] == "image":
                    print(f"Image: {test_case['input']}")
                    print(f"Final Sentiment: {result.get('final_sentiment', 'N/A')}")
                    print(f"Final Confidence: {result.get('final_confidence', 0):.3f}")
                    
                    # Show breakdown
                    breakdown = result.get("analysis_breakdown", {})
                    if breakdown.get("extracted_text"):
                        print(f"Extracted Text: '{breakdown['extracted_text'][:100]}...'")
                    if breakdown.get("image_caption"):
                        print(f"Image Caption: '{breakdown['image_caption']}'")
                    
                    # Show individual analyses
                    text_sentiment = breakdown.get("text_sentiment")
                    if text_sentiment and "sentiment" in text_sentiment:
                        print(f"Text Sentiment: {text_sentiment['sentiment']} ({text_sentiment['confidence']:.3f})")
                    
                    visual_sentiment = breakdown.get("visual_sentiment")
                    if visual_sentiment and "sentiment" in visual_sentiment:
                        print(f"Visual Sentiment: {visual_sentiment['sentiment']} ({visual_sentiment['confidence']:.3f})")
                
                print("✅ Analysis completed")
                results.append({
                    "test_case": test_case,
                    "result": result,
                    "status": "success"
                })
                
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
                results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 Test Summary")
        print("=" * 60)
        
        successful_tests = sum(1 for r in results if r["status"] == "success")
        total_tests = len(results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
        
        # Save detailed results
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful": successful_tests,
                "failed": total_tests - successful_tests,
                "success_rate": successful_tests/total_tests*100
            },
            "test_results": results
        }
        
        with open("multimodal_test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed test report saved to: multimodal_test_report.json")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Multimodal system is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the results above for details.")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"❌ Critical error in testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_capabilities():
    """Demonstrate the capabilities of the multimodal system"""
    
    print("\n🚀 Multimodal Meme Sentiment Analysis Capabilities")
    print("=" * 60)
    
    capabilities = [
        {
            "feature": "Text Analysis",
            "description": "Analyzes sentiment from pure text memes and captions",
            "methods": ["Trained BERT-based model", "Context understanding", "Emoji processing"]
        },
        {
            "feature": "Image Analysis", 
            "description": "Comprehensive image analysis with multiple approaches",
            "methods": [
                "OCR text extraction and sentiment",
                "Visual sentiment via CLIP model", 
                "Image captioning with BLIP",
                "Face detection and emotion analysis",
                "Multi-modal result fusion"
            ]
        },
        {
            "feature": "Video Analysis",
            "description": "Frame-by-frame video analysis with temporal aggregation",
            "methods": [
                "Key frame extraction",
                "Individual frame analysis",
                "Temporal sentiment aggregation",
                "Scene change detection"
            ]
        },
        {
            "feature": "Advanced Features",
            "description": "Additional capabilities for comprehensive analysis",
            "methods": [
                "Auto input type detection",
                "Confidence scoring",
                "Meme categorization",
                "Multi-modal fusion",
                "Batch processing support"
            ]
        }
    ]
    
    for capability in capabilities:
        print(f"\n🔧 {capability['feature']}")
        print(f"Description: {capability['description']}")
        print("Methods:")
        for method in capability['methods']:
            print(f"  • {method}")
    
    print(f"\n💡 Usage Examples:")
    print("=" * 40)
    print("# Text meme")
    print("analyzer.analyze_meme('This meme is hilarious!', 'text')")
    print("\n# Image meme")
    print("analyzer.analyze_meme('path/to/meme.jpg', 'image')")
    print("\n# Video meme")
    print("analyzer.analyze_meme('path/to/meme.mp4', 'video')")
    print("\n# Auto-detection")
    print("analyzer.analyze_meme('path/to/content')  # Auto-detects type")

if __name__ == "__main__":
    # Demonstrate capabilities first
    demonstrate_capabilities()
    
    # Run tests
    print("\n" + "=" * 60)
    success = test_multimodal_system()
    
    if success:
        print("\n🎉 Multimodal system is ready for use!")
    else:
        print("\n⚠️ Please check the setup and install missing dependencies.")
        print("Run: python install_multimodal_requirements.py")
