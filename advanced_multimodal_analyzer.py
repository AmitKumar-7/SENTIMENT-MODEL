#!/usr/bin/env python3
"""
Advanced Multimodal Meme Sentiment Analyzer
Handles images, videos, and text with visual context understanding
"""

import os
import cv2
import torch
import numpy as np
import requests
from PIL import Image, ImageEnhance
import pytesseract
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BlipProcessor, BlipForConditionalGeneration,
    AutoImageProcessor, AutoModelForImageClassification,
    CLIPProcessor, CLIPModel
)
import torchvision.transforms as transforms
import validators
from urllib.parse import urlparse
import tempfile
import json
from datetime import datetime
import logging
from typing import Dict, List, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Video processing
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not available. Video analysis will be limited.")

# Audio processing for video memes
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not available. Audio analysis disabled.")

from config import TESSERACT_PATH, PRODUCTION_MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMultimodalAnalyzer:
    """
    Advanced multimodal meme sentiment analyzer with visual understanding
    """
    
    def __init__(self, text_model_path: str = None):
        """
        Initialize the advanced multimodal analyzer
        
        Args:
            text_model_path: Path to the trained text sentiment model
        """
        self.text_model_path = text_model_path or PRODUCTION_MODEL_PATH
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]
        
        # Meme-specific sentiment indicators
        self.visual_sentiment_keywords = {
            "positive": [
                "happy", "smiling", "laughing", "celebration", "party", "success",
                "joy", "excited", "thumbs up", "heart", "love", "cute", "adorable",
                "funny", "hilarious", "amazing", "awesome", "perfect", "beautiful"
            ],
            "negative": [
                "sad", "crying", "angry", "frustrated", "disappointed", "upset",
                "mad", "furious", "depressed", "worried", "scared", "anxious",
                "terrible", "awful", "horrible", "disgusting", "hate", "annoying"
            ],
            "neutral": [
                "neutral", "calm", "serious", "thinking", "confused", "wondering",
                "observing", "standing", "sitting", "looking", "normal", "regular"
            ]
        }
        
        # Configure OCR
        if os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        
        # Load all models
        self.load_models()
        
    def load_models(self):
        """Load all required models for multimodal analysis"""
        try:
            logger.info("Loading multimodal models...")
            
            # 1. Text sentiment model (your trained model)
            logger.info("Loading text sentiment model...")
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_model_path)
            self.text_model.eval()
            
            # 2. Image captioning model (BLIP)
            logger.info("Loading image captioning model...")
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # 3. CLIP model for image-text similarity
            logger.info("Loading CLIP model...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # 4. Emotion recognition model
            logger.info("Loading emotion recognition model...")
            try:
                self.emotion_processor = AutoImageProcessor.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
                self.emotion_model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
            except:
                logger.warning("Could not load emotion model, using fallback")
                self.emotion_model = None
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise Exception(f"Could not load required models: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment from text using trained model
        """
        try:
            if not text or text.strip() == "":
                return {"sentiment": "Neutral", "confidence": 0.33, "method": "fallback"}
            
            # Tokenize text
            inputs = self.text_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.text_model(**inputs)
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
                },
                "method": "text_model"
            }
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {"error": str(e)}
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Enhanced text extraction with image preprocessing
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Multiple OCR attempts with different configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 8',  # Single word
                '--psm 13', # Raw line. Treat the image as a single text line
                '--psm 3'   # Default
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                except:
                    continue
            
            return best_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def generate_image_caption(self, image_path: str) -> str:
        """
        Generate descriptive caption for image using BLIP
        """
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate caption
            inputs = self.caption_processor(image, return_tensors="pt")
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return ""
    
    def analyze_visual_sentiment_clip(self, image_path: str) -> Dict:
        """
        Analyze visual sentiment using CLIP model
        """
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Sentiment prompts for CLIP
            sentiment_prompts = [
                "a funny and humorous meme",
                "a positive and happy meme", 
                "a wholesome and heartwarming meme",
                "a sad and depressing meme",
                "an angry and frustrated meme",
                "a sarcastic and cynical meme",
                "a neutral and informational meme",
                "a confused and questioning meme"
            ]
            
            # Process image and text
            inputs = self.clip_processor(
                text=sentiment_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Get similarity scores
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Map to sentiment categories
            sentiment_scores = {
                "Positive": float(probs[0][0] + probs[0][1] + probs[0][2]),  # funny, positive, wholesome
                "Negative": float(probs[0][3] + probs[0][4] + probs[0][5]),  # sad, angry, sarcastic
                "Neutral": float(probs[0][6] + probs[0][7])                  # neutral, confused
            }
            
            # Get highest scoring sentiment
            predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[predicted_sentiment]
            
            return {
                "sentiment": predicted_sentiment,
                "confidence": confidence,
                "all_scores": sentiment_scores,
                "method": "clip_visual"
            }
        except Exception as e:
            logger.error(f"Error analyzing visual sentiment with CLIP: {e}")
            return {"error": str(e)}
    
    def detect_faces_and_emotions(self, image_path: str) -> Dict:
        """
        Detect faces and emotions in image
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {"faces_detected": 0, "emotions": []}
            
            emotions = []
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                
                # Simple emotion heuristics based on image properties
                # (This is a simplified version - for production, use a proper emotion model)
                brightness = np.mean(face_roi)
                emotions.append({
                    "emotion": "positive" if brightness > 100 else "negative",
                    "confidence": 0.6
                })
            
            return {
                "faces_detected": len(faces),
                "emotions": emotions
            }
        except Exception as e:
            logger.error(f"Error detecting faces/emotions: {e}")
            return {"faces_detected": 0, "emotions": []}
    
    def analyze_image_comprehensive(self, image_path: str) -> Dict:
        """
        Comprehensive image analysis combining multiple methods
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # 1. Extract text via OCR
            extracted_text = self.extract_text_from_image(image_path)
            
            # 2. Generate image caption
            caption = self.generate_image_caption(image_path)
            
            # 3. Visual sentiment analysis
            visual_sentiment = self.analyze_visual_sentiment_clip(image_path)
            
            # 4. Face/emotion detection
            face_emotions = self.detect_faces_and_emotions(image_path)
            
            # 5. Text sentiment analysis (if text found)
            text_sentiment = None
            if extracted_text:
                text_sentiment = self.analyze_text_sentiment(extracted_text)
            
            # 6. Caption sentiment analysis
            caption_sentiment = None
            if caption:
                caption_sentiment = self.analyze_text_sentiment(caption)
            
            # Combine all analyses for final sentiment
            final_sentiment = self.combine_multimodal_results(
                text_sentiment, visual_sentiment, caption_sentiment, face_emotions
            )
            
            return {
                "final_sentiment": final_sentiment["sentiment"],
                "final_confidence": final_sentiment["confidence"],
                "analysis_breakdown": {
                    "extracted_text": extracted_text,
                    "image_caption": caption,
                    "text_sentiment": text_sentiment,
                    "visual_sentiment": visual_sentiment,
                    "caption_sentiment": caption_sentiment,
                    "face_emotions": face_emotions
                },
                "input_type": "image",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive image analysis: {e}")
            return {"error": str(e)}
    
    def analyze_video_comprehensive(self, video_path: str, max_frames: int = 10) -> Dict:
        """
        Comprehensive video analysis
        """
        try:
            if not MOVIEPY_AVAILABLE:
                return {"error": "MoviePy not available for video analysis"}
            
            logger.info(f"Analyzing video: {video_path}")
            
            # Extract key frames from video
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            # Extract frames at regular intervals
            frame_times = np.linspace(0, duration, min(max_frames, int(duration) + 1))
            frame_analyses = []
            
            temp_dir = tempfile.mkdtemp()
            
            for i, time_point in enumerate(frame_times):
                try:
                    # Extract frame
                    frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
                    clip.save_frame(frame_path, t=time_point)
                    
                    # Analyze frame
                    frame_analysis = self.analyze_image_comprehensive(frame_path)
                    frame_analysis["timestamp"] = time_point
                    frame_analyses.append(frame_analysis)
                    
                    # Clean up frame
                    os.remove(frame_path)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing frame at {time_point}s: {e}")
                    continue
            
            # Clean up
            clip.close()
            os.rmdir(temp_dir)
            
            # Aggregate frame analyses
            if not frame_analyses:
                return {"error": "No frames could be analyzed"}
            
            # Calculate overall sentiment
            overall_sentiment = self.aggregate_video_sentiment(frame_analyses)
            
            return {
                "final_sentiment": overall_sentiment["sentiment"],
                "final_confidence": overall_sentiment["confidence"],
                "video_duration": duration,
                "frames_analyzed": len(frame_analyses),
                "frame_analyses": frame_analyses,
                "input_type": "video",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            return {"error": str(e)}
    
    def combine_multimodal_results(self, text_result, visual_result, caption_result, face_emotions) -> Dict:
        """
        Combine results from different modalities
        """
        try:
            sentiments = []
            confidences = []
            
            # Add text sentiment if available
            if text_result and "sentiment" in text_result:
                sentiments.append(text_result["sentiment"])
                confidences.append(text_result["confidence"] * 0.4)  # 40% weight
            
            # Add visual sentiment
            if visual_result and "sentiment" in visual_result:
                sentiments.append(visual_result["sentiment"])
                confidences.append(visual_result["confidence"] * 0.4)  # 40% weight
            
            # Add caption sentiment if available
            if caption_result and "sentiment" in caption_result:
                sentiments.append(caption_result["sentiment"])
                confidences.append(caption_result["confidence"] * 0.2)  # 20% weight
            
            if not sentiments:
                return {"sentiment": "Neutral", "confidence": 0.33}
            
            # Weight-based voting
            sentiment_scores = {"Positive": 0, "Negative": 0, "Neutral": 0}
            
            for sentiment, confidence in zip(sentiments, confidences):
                sentiment_scores[sentiment] += confidence
            
            # Get final sentiment
            final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            final_confidence = sentiment_scores[final_sentiment] / sum(confidences) if sum(confidences) > 0 else 0.33
            
            return {
                "sentiment": final_sentiment,
                "confidence": min(final_confidence, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error combining multimodal results: {e}")
            return {"sentiment": "Neutral", "confidence": 0.33}
    
    def aggregate_video_sentiment(self, frame_analyses: List[Dict]) -> Dict:
        """
        Aggregate sentiment across video frames
        """
        try:
            sentiments = []
            confidences = []
            
            for frame in frame_analyses:
                if "final_sentiment" in frame:
                    sentiments.append(frame["final_sentiment"])
                    confidences.append(frame.get("final_confidence", 0.33))
            
            if not sentiments:
                return {"sentiment": "Neutral", "confidence": 0.33}
            
            # Calculate weighted average
            sentiment_scores = {"Positive": 0, "Negative": 0, "Neutral": 0}
            
            for sentiment, confidence in zip(sentiments, confidences):
                sentiment_scores[sentiment] += confidence
            
            final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            final_confidence = sentiment_scores[final_sentiment] / len(sentiments)
            
            return {
                "sentiment": final_sentiment,
                "confidence": min(final_confidence, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating video sentiment: {e}")
            return {"sentiment": "Neutral", "confidence": 0.33}
    
    def analyze_meme(self, input_data: str, input_type: str = "auto") -> Dict:
        """
        Main function to analyze meme sentiment from any input type
        """
        try:
            # Auto-detect input type if not specified
            if input_type == "auto":
                input_type = self.detect_input_type(input_data)
            
            logger.info(f"Analyzing {input_type} input: {input_data[:100]}...")
            
            if input_type == "text":
                result = self.analyze_text_sentiment(input_data)
                result["input_type"] = "text"
                
            elif input_type == "image":
                result = self.analyze_image_comprehensive(input_data)
                
            elif input_type == "video":
                result = self.analyze_video_comprehensive(input_data)
                
            else:
                return {"error": f"Unsupported input type: {input_type}"}
            
            # Add categorization
            if "final_sentiment" in result:
                result["meme_categories"] = self.categorize_meme(result["final_sentiment"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing meme: {e}")
            return {"error": str(e)}
    
    def detect_input_type(self, input_data: str) -> str:
        """
        Auto-detect input type
        """
        if os.path.isfile(input_data):
            ext = os.path.splitext(input_data)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                return "image"
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return "video"
        
        return "text"
    
    def categorize_meme(self, sentiment: str) -> List[str]:
        """
        Categorize meme based on sentiment
        """
        categories = {
            "Positive": ["Funny", "Wholesome", "Motivational"],
            "Negative": ["Sarcastic", "Dark Humor", "Criticism"],
            "Neutral": ["Observational", "Informational", "Factual"]
        }
        
        return categories.get(sentiment, ["Observational"])

def main():
    """Test the advanced multimodal analyzer"""
    analyzer = AdvancedMultimodalAnalyzer()
    
    # Test with sample text
    result = analyzer.analyze_meme("This meme is absolutely hilarious! 😂", "text")
    print("Text analysis:", json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
