#!/usr/bin/env python3
"""
Web Interface for Meme Sentiment Analyzer
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import tempfile
import logging
from werkzeug.utils import secure_filename
from meme_sentiment_analyzer import MemeSentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize analyzer
analyzer = MemeSentimentAnalyzer()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        result = {}
        
        # Handle different input types
        if 'text' in request.form and request.form['text'].strip():
            # Text analysis
            text = request.form['text'].strip()
            result = analyzer.analyze_meme(text, "text")
            
        elif 'url' in request.form and request.form['url'].strip():
            # URL analysis
            url = request.form['url'].strip()
            result = analyzer.analyze_meme(url, "url")
            
        elif 'file' in request.files:
            # File upload analysis
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = analyzer.analyze_meme(filepath, "auto")
                
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except:
                    pass
            else:
                result = {"error": "Invalid file type"}
        else:
            result = {"error": "No input provided"}
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        if not data or 'inputs' not in data:
            return jsonify({"error": "No inputs provided"}), 400
        
        results = analyzer.batch_analyze(data['inputs'])
        return jsonify({"results": results})
    
    except Exception as e:
        logger.error(f"Error in batch endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": hasattr(analyzer, 'model')})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create basic HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meme Sentiment Analyzer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 20px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .input-section h3 {
            margin-top: 0;
            color: #555;
        }
        input[type="text"], input[type="url"], textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            min-height: 100px;
        }
        .result.success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .sentiment-positive { color: #28a745; }
        .sentiment-negative { color: #dc3545; }
        .sentiment-neutral { color: #6c757d; }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 15px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            cursor: pointer;
            text-align: center;
            transition: background-color 0.3s;
        }
        .tab:first-child {
            border-radius: 5px 0 0 5px;
        }
        .tab:last-child {
            border-radius: 0 5px 5px 0;
        }
        .tab.active {
            background-color: #007bff;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Meme Sentiment Analyzer</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Analyze sentiment from memes in text, image, video, or URL format
        </p>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('text')">📝 Text</div>
            <div class="tab" onclick="showTab('file')">📁 File Upload</div>
            <div class="tab" onclick="showTab('url')">🔗 URL</div>
        </div>
        
        <form id="analyzeForm">
            <div id="text-tab" class="tab-content active">
                <div class="input-section">
                    <h3>📝 Text Analysis</h3>
                    <textarea name="text" placeholder="Enter meme text here..." rows="4"></textarea>
                </div>
            </div>
            
            <div id="file-tab" class="tab-content">
                <div class="input-section">
                    <h3>📁 File Upload</h3>
                    <input type="file" name="file" accept=".jpg,.jpeg,.png,.gif,.bmp,.mp4,.avi,.mov,.wmv">
                    <small style="color: #666;">Supported formats: Images (JPG, PNG, GIF, BMP) and Videos (MP4, AVI, MOV, WMV)</small>
                </div>
            </div>
            
            <div id="url-tab" class="tab-content">
                <div class="input-section">
                    <h3>🔗 URL Analysis</h3>
                    <input type="url" name="url" placeholder="https://example.com/meme.jpg">
                </div>
            </div>
            
            <button type="submit">🔍 Analyze Sentiment</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Clear form
            document.getElementById('analyzeForm').reset();
            document.getElementById('result').style.display = 'none';
        }
        
        document.getElementById('analyzeForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';
            resultDiv.innerHTML = '<div class="loading">🔄 Analyzing...</div>';
            
            const formData = new FormData(this);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>❌ Error</h3><p>${result.error}</p>`;
                } else {
                    resultDiv.className = 'result success';
                    let html = '<h3>✅ Analysis Results</h3>';
                    
                    if (result.sentiment) {
                        const sentimentClass = `sentiment-${result.sentiment.toLowerCase()}`;
                        const emoji = result.sentiment === 'Positive' ? '😊' : 
                                     result.sentiment === 'Negative' ? '😞' : '😐';
                        html += `<p><strong>Sentiment:</strong> <span class="${sentimentClass}">${emoji} ${result.sentiment}</span></p>`;
                        html += `<p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>`;
                    }
                    
                    if (result.meme_categories && result.meme_categories.length > 0) {
                        html += `<p><strong>Meme Categories:</strong> ${result.meme_categories.join(', ')}</p>`;
                    }
                    
                    if (result.extracted_text) {
                        html += `<p><strong>Extracted Text:</strong> "${result.extracted_text}"</p>`;
                    }
                    
                    if (result.all_scores) {
                        html += '<h4>Detailed Scores:</h4><ul>';
                        for (const [sentiment, score] of Object.entries(result.all_scores)) {
                            html += `<li>${sentiment}: ${(score * 100).toFixed(1)}%</li>`;
                        }
                        html += '</ul>';
                    }
                    
                    resultDiv.innerHTML = html;
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `<h3>❌ Error</h3><p>Network error: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
"""
    
    # Write the HTML template
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print("🚀 Starting Meme Sentiment Analyzer Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    # Set debug mode based on environment variable for security
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='127.0.0.1', port=5000)
