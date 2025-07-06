#!/usr/bin/env python3
"""
Sarcasm-Aware CLI for Multimodal Meme Sentiment Analysis
Enhanced CLI that properly handles sarcastic content
"""

import argparse
import json
import os
import sys
from datetime import datetime
from sarcasm_aware_analyzer import SarcasmAwareAnalyzer

def format_sarcasm_output(result, format_type="pretty"):
    """Format the analysis output with sarcasm information"""
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    if format_type == "json":
        return json.dumps(result, indent=2, default=str)
    
    # Pretty format with sarcasm awareness
    output = []
    output.append("🎯 Sarcasm-Aware Meme Sentiment Analysis")
    output.append("=" * 50)
    
    # Main results
    if "final_sentiment" in result:
        # Image/Video results
        sentiment = result["final_sentiment"]
        confidence = result.get("final_confidence", 0)
        
        # Add emoji based on sentiment
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "❓")
        
        output.append(f"Final Sentiment: {emoji} {sentiment}")
        output.append(f"Final Confidence: {confidence:.1%}")
        
        # Show sarcasm detection for images
        if result.get("sarcasm_detected_in_image"):
            sarcasm_score = result.get("image_sarcasm_score", 0)
            output.append(f"🎭 Sarcasm Detected in Image Text (score: {sarcasm_score:.2f})")
            if result.get("adjustment_reason"):
                output.append(f"   Adjustment: {result['adjustment_reason']}")
        
        # Show analysis breakdown
        breakdown = result.get("analysis_breakdown", {})
        
        if breakdown.get("extracted_text"):
            output.append(f"\n📝 Extracted Text:")
            output.append(f'   "{breakdown["extracted_text"][:100]}..."')
        
        if breakdown.get("image_caption"):
            output.append(f"\n🖼️  Image Caption:")
            output.append(f'   "{breakdown["image_caption"]}"')
        
        # Individual analysis results
        output.append(f"\n🔍 Analysis Breakdown:")
        
        if breakdown.get("text_sentiment") and "sentiment" in breakdown["text_sentiment"]:
            ts = breakdown["text_sentiment"]
            output.append(f"   📝 Text: {ts['sentiment']} ({ts['confidence']:.1%})")
        
        if breakdown.get("visual_sentiment") and "sentiment" in breakdown["visual_sentiment"]:
            vs = breakdown["visual_sentiment"] 
            output.append(f"   👁️  Visual: {vs['sentiment']} ({vs['confidence']:.1%})")
        
        if breakdown.get("caption_sentiment") and "sentiment" in breakdown["caption_sentiment"]:
            cs = breakdown["caption_sentiment"]
            output.append(f"   🖼️  Caption: {cs['sentiment']} ({cs['confidence']:.1%})")
        
        if breakdown.get("face_emotions"):
            faces = breakdown["face_emotions"]["faces_detected"]
            output.append(f"   👤 Faces Detected: {faces}")
    
    else:
        # Text-only results
        sentiment = result.get("sentiment", "Unknown")
        confidence = result.get("confidence", 0)
        
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "❓")
        
        output.append(f"Sentiment: {emoji} {sentiment}")
        output.append(f"Confidence: {confidence:.1%}")
        
        # Show sarcasm detection details
        if result.get("sarcasm_detected"):
            sarcasm_type = result.get("sarcasm_type", "Unknown")
            sarcasm_score = result.get("sarcasm_score", 0)
            original_sentiment = result.get("original_sentiment", "Unknown")
            
            output.append(f"\n🎭 Sarcasm Analysis:")
            output.append(f"   Type: {sarcasm_type}")
            output.append(f"   Score: {sarcasm_score:.2f}")
            output.append(f"   Original Classification: {original_sentiment}")
            
            if original_sentiment != sentiment:
                output.append(f"   ✅ Adjusted: {original_sentiment} → {sentiment}")
            else:
                output.append(f"   ➡️ Confirmed: {sentiment} (sarcasm acknowledged)")
        
        if "all_scores" in result:
            output.append(f"\nDetailed Scores:")
            scores = result["all_scores"]
            for label, score in scores.items():
                output.append(f"  {label}: {score:.1%}")
    
    # Add categories if available
    if "meme_categories" in result:
        output.append(f"\n🏷️  Meme Categories: {', '.join(result['meme_categories'])}")
    
    # Input type
    input_type = result.get("input_type", "unknown")
    output.append(f"\n📊 Input Type: {input_type.title()}")
    
    return "\n".join(output)

def main():
    """Main CLI function with sarcasm awareness"""
    
    parser = argparse.ArgumentParser(
        description="Sarcasm-Aware Multimodal Meme Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sarcastic text analysis
  python sarcasm_cli.py "Oh great, another Monday meme 🙄" --type text
  
  # Image analysis with sarcasm detection
  python sarcasm_cli.py "path/to/sarcastic_meme.jpg" --type image
  
  # Auto-detection with sarcasm awareness
  python sarcasm_cli.py "Wow, such original content 😏"
  
  # Compare with original model
  python sarcasm_cli.py "Perfect timing!" --compare
        """
    )
    
    parser.add_argument(
        "input",
        help="Input to analyze (text content, image path, or video path)"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["text", "image", "video", "auto"],
        default="auto",
        help="Input type (default: auto-detect)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Save results to file"
    )
    
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare with original analyzer"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize sarcasm-aware analyzer
    if args.verbose:
        print("🔄 Initializing sarcasm-aware analyzer...")
    
    try:
        analyzer = SarcasmAwareAnalyzer()
        if args.verbose:
            print("✅ Sarcasm-aware analyzer initialized!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Analyze input
    if args.verbose:
        print(f"🔄 Analyzing {args.type} input with sarcasm detection...")
    
    try:
        result = analyzer.analyze_meme(args.input, args.type)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)
    
    # Compare with original if requested
    if args.compare and args.type == "text":
        print("📊 Comparison Analysis")
        print("=" * 30)
        
        # Import original analyzer for comparison
        from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer
        original_analyzer = AdvancedMultimodalAnalyzer()
        
        original_result = original_analyzer.analyze_text_sentiment(args.input)
        
        print(f"Original Model:  {original_result.get('sentiment', 'Unknown')} ({original_result.get('confidence', 0):.1%})")
        print(f"Sarcasm-Aware:   {result.get('sentiment', 'Unknown')} ({result.get('confidence', 0):.1%})")
        
        if result.get('sarcasm_detected'):
            print(f"Sarcasm:         Detected ({result.get('sarcasm_type', 'Unknown')})")
        else:
            print(f"Sarcasm:         Not detected")
        
        print()
    
    # Format and display results
    formatted_output = format_sarcasm_output(result, args.format)
    print(formatted_output)
    
    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                if args.format == "json":
                    json.dump(result, f, indent=2, default=str)
                else:
                    f.write(formatted_output)
            print(f"\n📄 Results saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
    
    # Performance info
    if args.verbose:
        print(f"\n⚡ Analysis completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
