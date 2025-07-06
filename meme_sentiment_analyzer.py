#!/usr/bin/env python3
"""
Comprehensive Meme Sentiment Analyzer
Analyzes sentiment from memes in various formats: image, text, video, URL
"""

import os
import io
import cv2
import torch
import numpy as np
import requests
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import validators
from urllib.parse import urlparse
import tempfile
import json
from datetime import datetime
import logging
from typing import Dict, List, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import moviepy, make video processing optional
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        import moviepy
        from moviepy import VideoFileClip
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        print("Warning: MoviePy not available. Video analysis will be disabled.")

from config import TESSERACT_PATH, PRODUCTION_MODEL_PATH, DEFAULT_MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemeSentimentAnalyzer:
    """
    Comprehensive meme sentiment analyzer supporting multiple input formats
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the meme sentiment analyzer
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = model_path or PRODUCTION_MODEL_PATH
        self.sentiment_labels = {
            0: "Negative",
            1: "Neutral", 
            2: "Positive"
        }
        
        self.meme_categories = {
            "Negative": ["Dark Humor", "Sarcastic", "Criticism", "Disappointed", "Angry"],
            "Neutral": ["Observational", "Informational", "Questioning", "Factual", "Confused"],
            "Positive": ["Wholesome", "Funny", "Encouraging", "Celebratory", "Motivational"]
        }
        
        # Configure OCR
        if os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        
        # Load model and tokenizer
        self.load_model()
        
    def load_model(self):
        """Load the trained sentiment model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to default model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH)
                self.model.eval()
                logger.info("Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise Exception("Could not load any model")
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment from text content
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            sentiment = self.sentiment_labels[predicted_class]
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "all_scores": {
                    "Negative": predictions[0][0].item(),
                    "Neutral": predictions[0][1].item(),
                    "Positive": predictions[0][2].item()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {"error": str(e)}
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            # Open and process image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def analyze_image_sentiment(self, image_path: str) -> Dict:
        """
        Analyze sentiment from image (via OCR text extraction)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Extract text from image
            extracted_text = self.extract_text_from_image(image_path)
            
            if not extracted_text:
                return {
                    "error": "No text found in image",
                    "extracted_text": ""
                }
            
            # Analyze sentiment of extracted text
            sentiment_result = self.analyze_text_sentiment(extracted_text)
            sentiment_result["extracted_text"] = extracted_text
            sentiment_result["input_type"] = "image"
            
            return sentiment_result
        except Exception as e:
            logger.error(f"Error analyzing image sentiment: {e}")
            return {"error": str(e)}
    
    def extract_frames_from_video(self, video_path: str, num_frames: int = 5) -> List[str]:
        """
        Extract frames from video for analysis
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of extracted frame file paths
        """
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy not available. Cannot process video files.")
            return []
        
        try:
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp()
            frame_paths = []
            
            # Load video
            video = VideoFileClip(video_path)
            duration = video.duration
            
            # Extract frames at regular intervals
            for i in range(num_frames):
                time_point = (duration / num_frames) * i
                frame = video.get_frame(time_point)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
                frame_image = Image.fromarray(frame.astype('uint8'))
                frame_image.save(frame_path)
                frame_paths.append(frame_path)
            
            video.close()
            return frame_paths
        except Exception as e:
            logger.error(f"Error extracting frames from video: {e}")
            return []
    
    def analyze_video_sentiment(self, video_path: str) -> Dict:
        """
        Analyze sentiment from video (via frame extraction and OCR)
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not MOVIEPY_AVAILABLE:
            return {"error": "MoviePy not available. Video analysis is disabled."}
        
        try:
            # Extract frames from video
            frame_paths = self.extract_frames_from_video(video_path)
            
            if not frame_paths:
                return {"error": "Could not extract frames from video"}
            
            # Analyze sentiment from each frame
            frame_results = []
            all_text = []
            
            for i, frame_path in enumerate(frame_paths):
                frame_text = self.extract_text_from_image(frame_path)
                if frame_text:
                    all_text.append(frame_text)
                    frame_result = self.analyze_text_sentiment(frame_text)
                    frame_result["frame_index"] = i
                    frame_result["extracted_text"] = frame_text
                    frame_results.append(frame_result)
            
            # Clean up temporary files
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            
            if not frame_results:
                return {"error": "No text found in video frames"}
            
            # Aggregate results
            combined_text = " ".join(all_text)
            overall_sentiment = self.analyze_text_sentiment(combined_text)
            
            return {
                "overall_sentiment": overall_sentiment["sentiment"],
                "overall_confidence": overall_sentiment["confidence"],
                "combined_text": combined_text,
                "frame_analysis": frame_results,
                "input_type": "video"
            }
        except Exception as e:
            logger.error(f"Error analyzing video sentiment: {e}")
            return {"error": str(e)}
    
    def download_from_url(self, url: str) -> str:
        """
        Download content from URL
        
        Args:
            url: URL to download from
            
        Returns:
            Path to downloaded file
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from content-type or URL
            content_type = response.headers.get('content-type', '')
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1]
            
            if 'image' in content_type or file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                file_extension = '.jpg'
            elif 'video' in content_type or file_extension.lower() in ['.mp4', '.avi', '.mov', '.wmv']:
                file_extension = '.mp4'
            else:
                file_extension = '.jpg'  # Default to image
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading from URL: {e}")
            return None
    
    def analyze_url_sentiment(self, url: str) -> Dict:
        """
        Analyze sentiment from URL content
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Download content
            file_path = self.download_from_url(url)
            
            if not file_path:
                return {"error": "Could not download content from URL"}
            
            # Determine file type and analyze accordingly
            if file_path.endswith('.mp4'):
                result = self.analyze_video_sentiment(file_path)
            else:
                result = self.analyze_image_sentiment(file_path)
            
            result["source_url"] = url
            
            # Clean up temporary file
            try:
                os.remove(file_path)
            except:
                pass
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing URL sentiment: {e}")
            return {"error": str(e)}
    
    def categorize_meme(self, sentiment: str, text: str = "") -> List[str]:
        """
        Categorize meme based on sentiment and content
        
        Args:
            sentiment: Predicted sentiment
            text: Text content for additional context
            
        Returns:
            List of possible meme categories
        """
        categories = self.meme_categories.get(sentiment, [])
        
        # Add contextual categorization based on text content
        text_lower = text.lower()
        
        if sentiment == "Positive":
            if any(word in text_lower for word in ["wholesome", "love", "heart", "cute", "sweet"]):
                categories.insert(0, "Wholesome")
            elif any(word in text_lower for word in ["funny", "lol", "haha", "joke", "humor"]):
                categories.insert(0, "Funny")
            elif any(word in text_lower for word in ["you can", "believe", "strong", "achieve"]):
                categories.insert(0, "Motivational")
        
        elif sentiment == "Negative":
            if any(word in text_lower for word in ["sarcasm", "irony", "really", "obviously"]):
                categories.insert(0, "Sarcastic")
            elif any(word in text_lower for word in ["dark", "death", "pain", "suffering"]):
                categories.insert(0, "Dark Humor")
            elif any(word in text_lower for word in ["angry", "mad", "furious", "rage"]):
                categories.insert(0, "Angry")
        
        return categories[:3]  # Return top 3 categories
    
    def analyze_meme(self, input_data: str, input_type: str = "auto") -> Dict:
        """
        Main method to analyze meme sentiment
        
        Args:
            input_data: Input data (text, file path, or URL)
            input_type: Type of input ("text", "image", "video", "url", "auto")
            
        Returns:
            Complete analysis results
        """
        try:
            # Auto-detect input type if not specified
            if input_type == "auto":
                if validators.url(input_data):
                    input_type = "url"
                elif os.path.isfile(input_data):
                    ext = os.path.splitext(input_data)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                        input_type = "image"
                    elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
                        input_type = "video"
                    else:
                        input_type = "text"
                else:
                    input_type = "text"
            
            # Analyze based on input type
            if input_type == "text":
                result = self.analyze_text_sentiment(input_data)
                result["input_type"] = "text"
                result["content"] = input_data
            elif input_type == "image":
                result = self.analyze_image_sentiment(input_data)
            elif input_type == "video":
                result = self.analyze_video_sentiment(input_data)
            elif input_type == "url":
                result = self.analyze_url_sentiment(input_data)
            else:
                return {"error": f"Unsupported input type: {input_type}"}
            
            # Add meme categorization if sentiment analysis was successful
            if "sentiment" in result and "error" not in result:
                text_content = result.get("extracted_text", result.get("content", ""))
                result["meme_categories"] = self.categorize_meme(result["sentiment"], text_content)
                result["analysis_timestamp"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            logger.error(f"Error in analyze_meme: {e}")
            return {"error": str(e)}
    
    def batch_analyze(self, inputs: List[Dict]) -> List[Dict]:
        """
        Analyze multiple memes in batch
        
        Args:
            inputs: List of input dictionaries with 'data' and 'type' keys
            
        Returns:
            List of analysis results
        """
        results = []
        for i, input_item in enumerate(inputs):
            try:
                result = self.analyze_meme(
                    input_item["data"], 
                    input_item.get("type", "auto")
                )
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "error": str(e)
                })
        return results

def main():
    """
    Example usage of the MemeSentimentAnalyzer
    """
    analyzer = MemeSentimentAnalyzer()
    
    # Example analyses
    print("=== Meme Sentiment Analyzer ===\n")
    
    # Text analysis
    text_result = analyzer.analyze_meme("This meme is so funny! I can't stop laughing 😂", "text")
    print("Text Analysis:")
    print(json.dumps(text_result, indent=2))
    print()
    
    # Image analysis (if you have an image file)
    # image_result = analyzer.analyze_meme("path/to/your/meme.jpg", "image")
    # print("Image Analysis:")
    # print(json.dumps(image_result, indent=2))

if __name__ == "__main__":
    main()
