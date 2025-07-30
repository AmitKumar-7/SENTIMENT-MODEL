#!/usr/bin/env python3
"""
Complete Meme Sentiment Model Training from Scratch
This script builds a high-quality sentiment analysis model from the ground up
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
import logging
import random
from datetime import datetime
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(42)

class MemeDatasetCreator:
    """Create a comprehensive, balanced dataset for meme sentiment analysis"""
    
    def __init__(self):
        self.label_mapping = {
            'positive': 'Positive',
            'negative': 'Negative', 
            'neutral': 'Neutral',
            'Positive': 'Positive',
            'Negative': 'Negative',
            'Neutral': 'Neutral',
            0: 'Negative',
            1: 'Neutral', 
            2: 'Positive'
        }
        
    def create_comprehensive_dataset(self):
        """Create a large, balanced dataset with diverse meme content"""
        print("📊 Creating Comprehensive Meme Dataset")
        print("=" * 60)
        
        # Load existing data if available
        existing_data = self.load_existing_data()
        
        # Generate synthetic meme data
        synthetic_data = self.generate_meme_data()
        
        # Create additional contextual data
        contextual_data = self.generate_contextual_data()
        
        # Combine all data sources
        all_data = existing_data + synthetic_data + contextual_data
        
        # Create DataFrame and clean
        df = pd.DataFrame(all_data)
        df = self.clean_and_validate_data(df)
        
        # Balance the dataset
        df_balanced = self.balance_dataset(df)
        
        print(f"✅ Created comprehensive dataset with {len(df_balanced)} samples")
        return df_balanced
    
    def load_existing_data(self):
        """Load and clean existing data from various sources"""
        existing_data = []
        
        # Try to load from various CSV files
        csv_files = ['dataset.csv', 'dataset_fixed.csv', 'dataset_combined.csv']
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    print(f"📁 Loading existing data from {csv_file}")
                    
                    # Handle different column names
                    text_col = None
                    sentiment_col = None
                    
                    if 'text' in df.columns:
                        text_col = 'text'
                    elif 'extracted_text' in df.columns:
                        text_col = 'extracted_text'
                    
                    if 'sentiment' in df.columns:
                        sentiment_col = 'sentiment'
                    elif 'label' in df.columns:
                        sentiment_col = 'label'
                    
                    if text_col and sentiment_col:
                        for _, row in df.iterrows():
                            text = str(row[text_col]).strip()
                            sentiment = self.label_mapping.get(row[sentiment_col], 'Neutral')
                            
                            if text and text != 'No Text' and len(text) > 5:
                                existing_data.append({
                                    'text': text,
                                    'sentiment': sentiment,
                                    'source': csv_file
                                })
                    
                except Exception as e:
                    print(f"⚠️ Error loading {csv_file}: {e}")
        
        print(f"📥 Loaded {len(existing_data)} samples from existing data")
        return existing_data
    
    def generate_meme_data(self):
        """Generate diverse meme-specific training data"""
        meme_data = []
        
        # Positive meme examples with variety
        positive_memes = [
            # Humor and laughter
            "This meme is absolutely hilarious! 😂😂😂",
            "LMAO this made my day! So funny!",
            "I can't stop laughing at this! 🤣",
            "This is comedy gold right here!",
            "Best meme I've seen all week! 💯",
            "Dying of laughter! This is perfect! 😭😂",
            "This meme hits different! So good!",
            "Pure comedy genius! Love it! ⭐",
            "This is why I love the internet! 😂",
            "Absolutely brilliant meme! Well done! 👏",
            
            # Wholesome content
            "This wholesome meme made my heart warm ❤️",
            "Such a sweet and heartwarming post! 🥰",
            "This restored my faith in humanity!",
            "Wholesome content like this makes me smile 😊",
            "This is so pure and beautiful! Love it!",
            "Heartwarming meme! Thanks for sharing! 💕",
            "This wholesome energy is what we need! ✨",
            "Such a positive and uplifting meme! 🌟",
            "This made me feel all warm and fuzzy!",
            "Blessed to see such wholesome content! 🙏",
            
            # Achievement and celebration
            "Finally achieved my goal! Celebrating! 🎉",
            "When you succeed after trying so hard! YES! 🙌",
            "Victory tastes so sweet! Well deserved! 🏆",
            "Dreams do come true! So happy right now! ⭐",
            "Hard work pays off! Feeling amazing! 💪",
            "This win feels incredible! Celebrating! 🎊",
            "Success at last! Time to party! 🥳",
            "Achievement unlocked! Feeling proud! 🏅",
            "When everything goes perfectly! Amazing! ✨",
            "Living my best life right now! 🌈",
            
            # Love and affection
            "This meme shows so much love! Beautiful! 💖",
            "Spreading love and positivity! Heart full! ❤️",
            "This loving energy is everything! 💕",
            "Such a beautiful message of love! 🥰",
            "Love wins! This meme proves it! 💝",
            "Feeling all the love from this post! 💗",
            "This radiates pure love and joy! 💞",
            "Love this positive energy! Amazing! 💟",
            "Heart melting from this sweet content! 💘",
            "This love and kindness is beautiful! 💓"
        ]
        
        # Negative meme examples with variety
        negative_memes = [
            # Frustration and anger
            "I hate everything about this stupid situation!",
            "This is absolutely terrible! Worst ever! 😠",
            "Ugh, this makes me so angry! Horrible! 😡",
            "I'm so frustrated with this nonsense! 🤬",
            "This is infuriating! Can't stand it! 👎",
            "Absolutely awful! This sucks so much! 😤",
            "This terrible content ruins my day! Bad!",
            "I'm so mad about this! Disgusting! 🤮",
            "This is pathetic and disappointing! Ugh!",
            "Horrible! This makes me furious! 😾",
            
            # Sadness and depression
            "This makes me so sad and depressed 😭",
            "Feeling terrible after seeing this 😢",
            "This breaks my heart! So depressing! 💔",
            "Why does everything make me cry? 😿",
            "This sad content destroyed my mood 😞",
            "Feeling down and out after this 😔",
            "This depressing post ruined my day 😟",
            "So much sadness in this meme 😢",
            "This made me cry real tears 😭",
            "Heavy heart after reading this 💔",
            
            # Disgust and revulsion  
            "This is absolutely disgusting! Gross! 🤮",
            "Eww! This makes me sick! Terrible! 🤢",
            "Revolting content! I hate this! 😷",
            "This is so gross and disturbing! Yuck!",
            "Disgusted by this awful content! Bad!",
            "This makes my stomach turn! Horrible! 🤮",
            "Repulsive! This is terrible! Gross! 😝",
            "This disgusting post is awful! Hate it!",
            "Sickening content! This is bad! 🤢",
            "This revolting meme is terrible! Ugh!",
            
            # Disappointment and criticism
            "Such a disappointing and bad meme! Terrible!",
            "This is so poorly done! Disappointing! 👎",
            "Expected better! This is awful! Bad!",
            "What a letdown! This sucks! Horrible!",
            "Terrible execution! Very disappointing! 😒",
            "This fails on every level! Bad!",
            "Poor quality content! Not good! 👎",
            "This disappointing post is terrible!",
            "Badly done! This is disappointing! Ugh!",
            "This terrible content disappoints me!"
        ]
        
        # Neutral meme examples with variety
        neutral_memes = [
            # Days and time references
            "It's Wednesday my dudes! 🐸",
            "Today is Thursday, nothing special",
            "Another Monday morning routine",
            "Friday afternoon energy is here",
            "Sunday funday is upon us",
            "Tuesday vibes are in the air",
            "Saturday morning thoughts",
            "Midweek energy check",
            "Weekend preparation mode",
            "Regular weekday schedule",
            
            # Observations and statements
            "This is just a regular meme post",
            "Some random thoughts about things",
            "Here's some content for your timeline",
            "Just posting this for no reason",
            "Random meme content appearing here",
            "This exists in the internet space",
            "Standard social media post here",
            "Regular content for your feed",
            "Just another post among many",
            "This is happening right now",
            
            # Questions and wondering
            "What time is it right now? 🕐",
            "I wonder what's for lunch today?",
            "How is everyone doing today?",
            "What should I watch on TV? 📺",
            "Where did I put my keys? 🗝️",
            "What's the weather like outside? 🌤️",
            "How was your day today?",
            "What are you thinking about? 🤔",
            "Which option should I choose?",
            "What's happening in the news? 📰",
            
            # Factual statements
            "Water is made of hydrogen and oxygen",
            "There are 24 hours in a day",
            "The sky appears blue during daytime",
            "Plants need sunlight to grow",
            "Coffee is a popular morning beverage ☕",
            "Books contain written information",
            "Cars have four wheels typically",
            "Rain comes from clouds ☁️",
            "Computers process digital information",
            "Food provides energy for humans"
        ]
        
        # Add positive examples
        for text in positive_memes:
            meme_data.append({
                'text': text,
                'sentiment': 'Positive',
                'source': 'generated_positive'
            })
        
        # Add negative examples  
        for text in negative_memes:
            meme_data.append({
                'text': text,
                'sentiment': 'Negative',
                'source': 'generated_negative'
            })
        
        # Add neutral examples
        for text in neutral_memes:
            meme_data.append({
                'text': text,
                'sentiment': 'Neutral',
                'source': 'generated_neutral'
            })
        
        print(f"🎭 Generated {len(meme_data)} synthetic meme samples")
        return meme_data
    
    def generate_contextual_data(self):
        """Generate contextual and edge case data"""
        contextual_data = []
        
        # Emoji-only content
        emoji_data = [
            # Positive emojis
            ("😂🤣😄😁😊", "Positive"),
            ("❤️💕💖😍🥰", "Positive"), 
            ("🎉🎊🥳🙌👏", "Positive"),
            ("🔥💯⭐✨🌟", "Positive"),
            ("😎😋😜🤗😇", "Positive"),
            
            # Negative emojis
            ("😭😢😞😔😟", "Negative"),
            ("😠😡🤬😤👎", "Negative"),
            ("🤮🤢😷💀☠️", "Negative"),
            ("😰😨😱😖😣", "Negative"),
            ("💔😿😾🙄😒", "Negative"),
            
            # Neutral emojis
            ("🤔😐😑🙂😶", "Neutral"),
            ("🧐🤨😯😮😊", "Neutral"),
            ("📱💻📚🏠🚗", "Neutral"),
            ("🌍🌙⭐🌤️⛅", "Neutral"),
            ("📅⏰🗓️📋📝", "Neutral")
        ]
        
        # Internet slang and abbreviations
        slang_data = [
            # Positive slang
            ("OMG this is so lit! Fire content! 🔥", "Positive"),
            ("Bruh this slaps! No cap! 💯", "Positive"),
            ("This hits different! Chef's kiss! 👌", "Positive"),
            ("Absolute banger! This goes hard! 🎵", "Positive"),
            ("Periodt! This is a whole mood! ✨", "Positive"),
            
            # Negative slang
            ("This ain't it chief... big yikes", "Negative"),
            ("Cringe content alert! Hard pass", "Negative"),
            ("This is straight trash ngl 🗑️", "Negative"),
            ("Bruh this is mid at best... meh", "Negative"),
            ("Not it! This is so sus 😬", "Negative"),
            
            # Neutral slang
            ("It is what it is tbh 🤷", "Neutral"),
            ("Just vibing here... nothing special", "Neutral"),
            ("Random thoughts going brrrr", "Neutral"),
            ("Just another Tuesday mood", "Neutral"),
            ("Existing in the void rn", "Neutral")
        ]
        
        # Mixed case and punctuation
        mixed_data = [
            ("THIS IS AMAZING!!! LOVE IT!!!", "Positive"),
            ("absolutely terrible... hate this...", "Negative"),
            ("just... existing... here... today...", "Neutral"),
            ("WORST. DAY. EVER. UGH!!!", "Negative"),
            ("Best! Thing! Ever! Amazing!", "Positive"),
            ("meh... could be better... or worse...", "Neutral")
        ]
        
        # Add emoji data
        for text, sentiment in emoji_data:
            contextual_data.append({
                'text': text,
                'sentiment': sentiment,
                'source': 'emoji_context'
            })
        
        # Add slang data
        for text, sentiment in slang_data:
            contextual_data.append({
                'text': text,
                'sentiment': sentiment,
                'source': 'slang_context'
            })
        
        # Add mixed case data
        for text, sentiment in mixed_data:
            contextual_data.append({
                'text': text,
                'sentiment': sentiment,
                'source': 'mixed_context'
            })
        
        print(f"🎯 Generated {len(contextual_data)} contextual samples")
        return contextual_data
    
    def clean_and_validate_data(self, df):
        """Clean and validate the dataset"""
        print("🧹 Cleaning and validating dataset...")
        
        initial_count = len(df)
        
        # Remove invalid entries
        df = df.dropna(subset=['text', 'sentiment'])
        df = df[df['text'].str.len() > 2]  # Minimum length
        df = df[df['text'].str.len() < 1000]  # Maximum length
        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Clean text
        df['text'] = df['text'].str.strip()
        
        cleaned_count = len(df)
        print(f"🗑️ Removed {initial_count - cleaned_count} invalid samples")
        print(f"✅ {cleaned_count} clean samples remaining")
        
        return df.reset_index(drop=True)
    
    def balance_dataset(self, df):
        """Balance the dataset to have equal representation"""
        print("⚖️ Balancing dataset...")
        
        # Get class counts
        class_counts = df['sentiment'].value_counts()
        print("📊 Original distribution:")
        for sentiment, count in class_counts.items():
            print(f"  {sentiment}: {count} samples")
        
        # Find target size (average of all classes, with minimum of 200)
        target_size = max(200, int(class_counts.mean()))
        
        balanced_dfs = []
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            class_df = df[df['sentiment'] == sentiment]
            
            if len(class_df) < target_size:
                # Upsample with replacement
                upsampled = class_df.sample(n=target_size, replace=True, random_state=42)
                balanced_dfs.append(upsampled)
                print(f"📈 Upsampled {sentiment}: {len(class_df)} → {target_size}")
            else:
                # Downsample if too large
                downsampled = class_df.sample(n=target_size, random_state=42)
                balanced_dfs.append(downsampled)
                print(f"📉 Downsampled {sentiment}: {len(class_df)} → {target_size}")
        
        # Combine and shuffle
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Balanced dataset: {len(balanced_df)} total samples")
        return balanced_df

class MemeDataset(Dataset):
    """PyTorch Dataset for meme sentiment classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MemeModelTrainer:
    """Complete model trainer with best practices"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.label_to_id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.id_to_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.tokenizer = None
        self.model = None
        
    def prepare_data(self, df):
        """Prepare data for training with proper splits"""
        print("🔧 Preparing data for training...")
        
        # Convert labels to numeric
        df['label_id'] = df['sentiment'].map(self.label_to_id)
        
        # Create stratified train/validation/test split
        X = df['text'].values
        y = df['label_id'].values
        
        # First split: train (70%) and temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Second split: validation (15%) and test (15%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"📊 Data splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Testing: {len(X_test)} samples")
        
        # Print class distribution for each split
        for split_name, y_split in [("Training", y_train), ("Validation", y_val), ("Testing", y_test)]:
            dist = Counter(y_split)
            print(f"\n📈 {split_name} distribution:")
            for label_id, count in dist.items():
                label_name = self.id_to_label[label_id]
                percentage = count / len(y_split) * 100
                print(f"  {label_name}: {count} ({percentage:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"🤖 Setting up model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            label2id=self.label_to_id,
            id2label=self.id_to_label
        )
        
        print("✅ Model and tokenizer ready!")
    
    def compute_class_weights(self, y_train):
        """Compute class weights for imbalanced data"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"⚖️ Class weights:")
        for label_id, weight in class_weight_dict.items():
            label_name = self.id_to_label[label_id]
            print(f"  {label_name}: {weight:.3f}")
        
        return class_weight_dict
    
    def create_weighted_trainer(self, class_weights):
        """Create trainer with class weights"""
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # Apply class weights
                device = logits.device
                weight_tensor = torch.tensor(
                    [class_weights[i] for i in range(3)], 
                    dtype=torch.float32, 
                    device=device
                )
                
                loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
                loss = loss_fct(logits.view(-1, 3), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        return WeightedTrainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the model with best practices"""
        print("🚀 Starting model training...")
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Create datasets
        train_dataset = MemeDataset(X_train, y_train, self.tokenizer)
        val_dataset = MemeDataset(X_val, y_val, self.tokenizer)
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Training arguments with best practices
        training_args = TrainingArguments(
            output_dir='./meme_model_v3',
            num_train_epochs=8,  # More epochs for better learning
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./meme_model_v3/logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            learning_rate=3e-5,
            gradient_accumulation_steps=2,
            dataloader_drop_last=False,
            seed=42,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            report_to=None  # Disable wandb/tensorboard logging
        )
        
        # Create weighted trainer
        WeightedTrainer = self.create_weighted_trainer(class_weights)
        
        # Initialize trainer
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("🏃‍♂️ Training in progress...")
        trainer.train()
        
        # Save the final model
        final_model_path = './meme_sentiment_model_final'
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print(f"✅ Training completed!")
        print(f"📁 Model saved to: {final_model_path}")
        
        return trainer
    
    def evaluate_model(self, trainer, X_test, y_test):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model performance...")
        
        # Create test dataset
        test_dataset = MemeDataset(X_test, y_test, self.tokenizer)
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n🎯 Final Model Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        target_names = ['Negative', 'Neutral', 'Positive']
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, target_names)
        
        # Test with sample inputs
        self.test_sample_predictions()
        
        # Save evaluation results
        eval_results = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return accuracy, f1
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Meme Sentiment Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('meme_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Confusion matrix saved as: meme_model_confusion_matrix.png")
    
    def test_sample_predictions(self):
        """Test model with diverse sample inputs"""
        print(f"\n🧪 Testing Sample Predictions:")
        print("=" * 60)
        
        test_cases = [
            ("This meme is absolutely hilarious! 😂😂😂", "Positive"),
            ("I hate everything about this stupid day", "Negative"),
            ("It's Wednesday my dudes", "Neutral"),
            ("Best meme ever! Love it so much!", "Positive"),
            ("Ugh, another boring meeting", "Negative"),
            ("Today is Thursday", "Neutral"),
            ("This is amazing! 🔥💯", "Positive"),
            ("This sucks so much 👎", "Negative"),
            ("Just another regular day", "Neutral"),
            ("Love this wholesome content ❤️", "Positive"),
            ("Terrible meme! Not funny at all", "Negative"),
            ("Some random meme content", "Neutral"),
            ("LMAO this is so lit! 🔥", "Positive"),
            ("This is straight trash ngl", "Negative"),
            ("It is what it is tbh 🤷", "Neutral")
        ]
        
        correct = 0
        for text, expected in test_cases:
            # Tokenize and predict
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            predicted = self.id_to_label[predicted_class]
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} '{text[:50]}...'")
            print(f"   Predicted: {predicted} (Expected: {expected}) - Confidence: {confidence:.2%}")
        
        accuracy = correct / len(test_cases)
        print(f"\n📊 Sample Test Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
        
        return accuracy

def main():
    """Main training pipeline"""
    print("🚀 COMPLETE MEME SENTIMENT MODEL TRAINING")
    print("=" * 70)
    print("Building a high-quality sentiment analysis model from scratch")
    print()
    
    # Step 1: Create comprehensive dataset
    dataset_creator = MemeDatasetCreator()
    df = dataset_creator.create_comprehensive_dataset()
    
    # Save the dataset
    df.to_csv('meme_training_dataset_final.csv', index=False)
    print(f"💾 Dataset saved as: meme_training_dataset_final.csv")
    
    # Step 2: Initialize trainer
    trainer = MemeModelTrainer()
    
    # Step 3: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Step 4: Train model
    model_trainer = trainer.train_model(X_train, X_val, y_train, y_val)
    
    # Step 5: Evaluate model
    accuracy, f1 = trainer.evaluate_model(model_trainer, X_test, y_test)
    
    # Final results
    print(f"\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✅ Final Model Accuracy: {accuracy:.2%}")
    print(f"✅ Final Model F1-Score: {f1:.3f}")
    print(f"📁 Model Location: ./meme_sentiment_model_final")
    print(f"📊 Confusion Matrix: meme_model_confusion_matrix.png")
    print(f"📋 Evaluation Results: evaluation_results.json")
    print(f"💾 Training Dataset: meme_training_dataset_final.csv")
    
    if accuracy >= 0.80:
        print(f"\n🎉 EXCELLENT PERFORMANCE! Ready for production!")
    elif accuracy >= 0.70:
        print(f"\n✅ GOOD PERFORMANCE! Suitable for most applications!")
    elif accuracy >= 0.60:
        print(f"\n⚠️ FAIR PERFORMANCE! Consider collecting more data!")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT! Check data quality and model settings!")
    
    print(f"\n📋 Next Steps:")
    print("1. Test the model: python test_new_model.py")
    print("2. Update your analyzer to use the new model")
    print("3. Deploy to production")
    print("4. Monitor performance and collect feedback")
    
    return accuracy, f1

if __name__ == "__main__":
    main()
