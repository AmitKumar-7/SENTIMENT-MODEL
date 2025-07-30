# config.py
# Secure configuration using environment variables

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Custom Search API
API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key_here")
CSE_ID = os.getenv("GOOGLE_CSE_ID", "your_custom_search_engine_id_here")

# Reddit API  
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "your_reddit_client_id_here")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "your_reddit_client_secret_here")

# Tesseract OCR Path
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Model paths
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "./fine_tuned_model")
PRODUCTION_MODEL_PATH = os.getenv("PRODUCTION_MODEL_PATH", "./production_model_v2")
