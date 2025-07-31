# 🎯 Meme Sentiment Analyzer

A powerful multimodal sentiment analysis system that can analyze memes from various formats including text, images, videos, and URLs. The system uses state-of-the-art NLP and computer vision models to determine sentiment with high accuracy.

## ✨ Features

- **Advanced Multimodal Analysis**:
  - 📝 Text sentiment analysis with fine-tuned BERT model
  - 🖼️ Image analysis with enhanced OCR and visual understanding
  - 🎥 Video analysis with frame extraction and processing
  - 🔗 URL content analysis and processing
  - 🤖 CLIP and BLIP model integration for visual context

- **High-Performance Analysis**:
  - Sentiment classification (Positive, Negative, Neutral)
  - Confidence scoring with detailed breakdowns
  - Visual sentiment understanding
  - Emotion detection capabilities

- **Multiple Interfaces**:
  - 🚀 FastAPI server for production use
  - 🌐 Flask web interface for interactive analysis
  - 🐍 Python API for integration

## 🏗️ Project Structure

```
sentimentmodel_cleaned/
├── advanced_multimodal_analyzer.py  # Main multimodal analysis class
├── api_main.py                      # FastAPI server
├── web_app.py                       # Flask web interface
├── config.py                        # Configuration settings
├── train_from_scratch.py            # Model training script
├── test_single_image.py             # Testing utilities
├── requirements.txt                 # Python dependencies
├── requirements_api.txt             # API-specific dependencies
├── START_HERE.md                    # Training guide
├── meme_training_dataset_final.csv  # Training dataset
├── meme_sentiment_model_final/      # Trained model files
└── templates/                       # Web interface templates
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd sentimentmodel_cleaned

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup OCR (Required for image/video analysis)

**Windows:**
1. Download Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and note the installation path
3. Update the `TESSERACT_PATH` in `config.py` or set as environment variable:
```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### 3. Train Your Model (Recommended First Step)

```bash
# Train a high-quality model from scratch (see START_HERE.md for details)
python train_from_scratch.py
```

### 4. Basic Usage

#### Python API
```python
from advanced_multimodal_analyzer import AdvancedMultimodalAnalyzer

# Initialize analyzer
analyzer = AdvancedMultimodalAnalyzer()

# Analyze text
result = analyzer.analyze_text_sentiment("This meme is hilarious! 😂")
print(result)

# Analyze image
result = analyzer.analyze_meme("path/to/meme.jpg", "image")
print(result)
```

#### FastAPI Server
```bash
# Start FastAPI server for production
uvicorn api_main:app --reload

# API will be available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

#### Flask Web Interface
```bash
# Start Flask web server
python web_app.py

# Open browser and go to: http://localhost:5000
```

#### Test Your Setup
```bash
# Test with a single image
python test_single_image.py
```

## 🔧 Configuration

### Environment Variables (.env file)
```env
# Model paths
MODEL_PATH=./production_model_v2
PRODUCTION_MODEL_PATH=./production_model_v2

# OCR Configuration
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# API Keys (optional - for advanced features)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
```

## 📊 Model Information

The system uses a fine-tuned BERT model trained on meme-specific sentiment data:

- **Base Model**: BERT-base-uncased
- **Training Data**: Custom meme dataset with sentiment labels
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Performance**: High accuracy on meme-specific sentiment classification

### Available Models
- `production_model_v2/`: Latest production model (recommended)
- `fine_tuned_model/`: Alternative trained model
- `trained_model/`: Basic trained model

## 🎨 Meme Categories

The system categorizes memes based on sentiment and content:

### Positive Categories
- 🥰 **Wholesome**: Heartwarming, cute, loving content
- 😂 **Funny**: Humorous, joke-based content
- 💪 **Motivational**: Encouraging, inspirational content
- 🎉 **Celebratory**: Achievement, success-based content

### Negative Categories
- 😏 **Sarcastic**: Ironic, sarcastic content
- 🖤 **Dark Humor**: Dark, edgy humor
- 😠 **Angry**: Angry, frustrated content
- 😞 **Disappointed**: Disappointing, sad content
- 📝 **Criticism**: Critical, complaint-based content

### Neutral Categories
- 👀 **Observational**: Observing situations
- ℹ️ **Informational**: Sharing information
- ❓ **Questioning**: Asking questions
- 📋 **Factual**: Stating facts
- 😕 **Confused**: Expressing confusion

## 🔍 API Reference

### MemeSentimentAnalyzer Class

#### Methods

##### `analyze_meme(input_data, input_type="auto")`
Main analysis method supporting all input types.

**Parameters:**
- `input_data` (str): Input data (text, file path, or URL)
- `input_type` (str): Type of input ("text", "image", "video", "url", "auto")

**Returns:**
- Dictionary with analysis results including sentiment, confidence, categories

##### `analyze_text_sentiment(text)`
Analyze sentiment from text content.

##### `analyze_image_sentiment(image_path)`
Analyze sentiment from image using OCR.

##### `analyze_video_sentiment(video_path)`
Analyze sentiment from video frames.

##### `analyze_url_sentiment(url)`
Analyze sentiment from URL content.

##### `batch_analyze(inputs)`
Analyze multiple inputs in batch.

### Example Response
```json
{
  "sentiment": "Positive",
  "confidence": 0.92,
  "meme_categories": ["Funny", "Wholesome"],
  "extracted_text": "When you finally understand the meme",
  "input_type": "image",
  "all_scores": {
    "Negative": 0.03,
    "Neutral": 0.05,
    "Positive": 0.92
  },
  "analysis_timestamp": "2024-01-15T10:30:00"
}
```

## 🌐 Web Interface Features

The web interface provides:
- **Multi-tab Input**: Text, File Upload, URL tabs
- **Real-time Analysis**: Instant results with progress indicators
- **Rich Visualization**: Color-coded sentiment results with emojis
- **Responsive Design**: Works on desktop and mobile devices
- **Batch Processing**: API endpoint for batch analysis

### Web API Endpoints

- `GET /`: Main interface
- `POST /analyze`: Single analysis endpoint
- `POST /batch`: Batch analysis endpoint
- `GET /health`: Health check endpoint

## 🛠️ Development

### Running Tests
```bash
# Test with sample meme
python meme_sentiment_analyzer.py

# Test CLI
python meme_cli.py "Test meme text" --format pretty

# Test web interface
python web_app.py
```

### Adding New Features
1. Extend the `MemeSentimentAnalyzer` class
2. Add new categorization rules in `categorize_meme()`
3. Update the web interface for new features
4. Add tests for new functionality

## 📁 Dataset

The project includes:
- `dataset.csv`: Training dataset with text and sentiment labels
- `meme_dataset/`: Sample meme images for testing
- Training data with positive, negative, and neutral examples

## 🔮 Future Enhancements

- **Multi-language Support**: Support for memes in different languages
- **Advanced Visual Analysis**: Direct image sentiment analysis without OCR
- **Real-time Processing**: Stream processing capabilities
- **Social Media Integration**: Direct integration with social platforms
- **Emotion Detection**: More granular emotion classification
- **Trend Analysis**: Meme trend detection and analysis

## 🐛 Troubleshooting

### Common Issues

**OCR Not Working:**
- Ensure Tesseract is properly installed
- Check the `TESSERACT_PATH` in your `.env` file
- Verify image quality and text clarity

**Model Loading Errors:**
- Check if model files exist in the specified path
- Ensure all model dependencies are installed
- Try using the fallback model path

**Memory Issues:**
- Reduce batch size for large datasets
- Use CPU instead of GPU if memory limited
- Process videos with fewer frames

**Web Interface Issues:**
- Check if port 5000 is available
- Ensure Flask dependencies are installed
- Check browser console for JavaScript errors

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Include error logs and system information

---

**Happy Meme Analysis! 🎉**
