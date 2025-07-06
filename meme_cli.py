#!/usr/bin/env python3
"""
Command Line Interface for Meme Sentiment Analyzer
"""

import argparse
import json
import sys
import os
from pathlib import Path
from meme_sentiment_analyzer import MemeSentimentAnalyzer

def format_output(result, format_type="json"):
    """Format output for display"""
    if format_type == "json":
        return json.dumps(result, indent=2)
    elif format_type == "pretty":
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        output = []
        output.append("🎯 Meme Sentiment Analysis Results")
        output.append("=" * 40)
        
        if "sentiment" in result:
            sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
            emoji = sentiment_emoji.get(result["sentiment"], "🤔")
            output.append(f"Sentiment: {emoji} {result['sentiment']}")
            output.append(f"Confidence: {result['confidence']:.2%}")
        
        if "meme_categories" in result:
            output.append(f"Meme Categories: {', '.join(result['meme_categories'])}")
        
        if "extracted_text" in result:
            output.append(f"Extracted Text: {result['extracted_text'][:100]}...")
        
        if "input_type" in result:
            output.append(f"Input Type: {result['input_type']}")
        
        if "all_scores" in result:
            output.append("\nDetailed Scores:")
            for sentiment, score in result["all_scores"].items():
                output.append(f"  {sentiment}: {score:.2%}")
        
        return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Analyze meme sentiment from various input types")
    parser.add_argument("input", help="Input data (text, file path, or URL)")
    parser.add_argument("-t", "--type", choices=["auto", "text", "image", "video", "url"], 
                       default="auto", help="Input type (default: auto-detect)")
    parser.add_argument("-m", "--model", help="Path to model directory")
    parser.add_argument("-f", "--format", choices=["json", "pretty"], default="pretty",
                       help="Output format (default: pretty)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-b", "--batch", help="Batch analyze from JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = MemeSentimentAnalyzer(model_path=args.model)
        
        if args.batch:
            # Batch analysis
            with open(args.batch, 'r') as f:
                batch_inputs = json.load(f)
            results = analyzer.batch_analyze(batch_inputs)
            output = format_output(results, args.format)
        else:
            # Single analysis
            result = analyzer.analyze_meme(args.input, args.type)
            output = format_output(result, args.format)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results saved to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
