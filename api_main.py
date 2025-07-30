
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import shutil
import os
import logging
from typing import List, Dict, Any

# Add the current directory to the system path to find the analyzer
import sys
sys.path.append(os.getcwd())

from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a temporary directory for uploads
UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize the FastAPI app
app = FastAPI(
    title="Meme Sentiment Analysis API",
    description="A comprehensive API for analyzing meme sentiment using a multimodal model.",
    version="1.0.0",
)

# Load the analyzer model once at startup
analyzer = None

@app.on_event("startup")
def load_model():
    """Load the model when the API starts to avoid reloading on every request."""
    global analyzer
    logger.info("Loading the AdvancedMultimodalAnalyzer model...")
    try:
        analyzer = AdvancedMultimodalAnalyzer()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # You might want to prevent the app from starting if the model fails to load
        # For now, we'll log the error and let it continue

# --- Pydantic Models for Request and Response ---

class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    final_sentiment: str
    final_confidence: float
    meme_categories: List[str]
    analysis_breakdown: Dict[str, Any]

# --- API Endpoints ---

@app.get("/", summary="Health Check")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "Meme Analysis API is running!", "model_loaded": analyzer is not None}

@app.post("/analyze-image/", response_model=AnalysisResponse, summary="Analyze Meme Image")
async def analyze_image(file: UploadFile = File(..., description="Meme image file (JPG, PNG, etc.)")):
    """
    Uploads a meme image, analyzes it, and returns the sentiment analysis.
    - **Handles image uploads**
    - **Performs multimodal analysis** (text, visual, faces)
    - **Returns detailed results**
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load. Please check API logs.")

    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Analyzing image: {file.filename}")
        # Analyze the image using the analyzer
        result = analyzer.analyze_meme(temp_file_path, "image")
        return result

    except Exception as e:
        logger.error(f"Error analyzing image {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze-text/", response_model=AnalysisResponse, summary="Analyze Meme Text")
def analyze_text(request: TextRequest):
    """
    Analyzes raw text content and returns the sentiment analysis.
    - **Input**: A simple JSON with a "text" field.
    - **Returns**: Detailed sentiment analysis results.
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check API logs.")

    try:
        logger.info(f"Analyzing text: \"{request.text[:50]}...\"")
        result = analyzer.analyze_meme(request.text, "text")
        return result

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")


