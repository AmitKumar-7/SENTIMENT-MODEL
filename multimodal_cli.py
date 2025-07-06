#!/usr/bin/env python3
"""
Enhanced CLI for Multimodal Meme Sentiment Analysis
Supports text, image, and video analysis with detailed output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer

def format_output(result, format_type="pretty"):
    """Format the analysis output"""
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    if format_type == "json":
        return json.dumps(result, indent=2, default=str)
    
    # Pretty format
    output = []
    output.append("🎯 Multimodal Meme Sentiment Analysis Results")
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
        
        # Video specific info
        if result.get("input_type") == "video":
            output.append(f"\n🎥 Video Info:")
            output.append(f"   Duration: {result.get('video_duration', 0):.1f}s")
            output.append(f"   Frames Analyzed: {result.get('frames_analyzed', 0)}")
    
    else:
        # Text-only results
        sentiment = result.get("sentiment", "Unknown")
        confidence = result.get("confidence", 0)
        
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "❓")
        
        output.append(f"Sentiment: {emoji} {sentiment}")
        output.append(f"Confidence: {confidence:.1%}")
        
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
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Advanced Multimodal Meme Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text analysis
  python multimodal_cli.py "This meme is hilarious!" --type text
  
  # Image analysis  
  python multimodal_cli.py "path/to/meme.jpg" --type image
  
  # Video analysis
  python multimodal_cli.py "path/to/meme.mp4" --type video
  
  # Auto-detection
  python multimodal_cli.py "path/to/content"  # Auto-detects type
  
  # JSON output
  python multimodal_cli.py "Funny meme!" --format json
  
  # Save to file
  python multimodal_cli.py "meme.jpg" --output results.json
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
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    if args.verbose:
        print("🔄 Initializing multimodal analyzer...")
    
    try:
        analyzer = AdvancedMultimodalAnalyzer()
        if args.verbose:
            print("✅ Analyzer initialized!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Analyze input
    if args.verbose:
        print(f"🔄 Analyzing {args.type} input...")
    
    try:
        result = analyzer.analyze_meme(args.input, args.type)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)
    
    # Format and display results
    formatted_output = format_output(result, args.format)
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
