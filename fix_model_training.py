#!/usr/bin/env python3
"""
Fix Model Training - Address bias and accuracy issues
This script retrains your model with proper techniques to fix the current problems
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

class MemeDataset(Dataset):
    """Custom dataset for meme sentiment classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

class ImprovedModelTrainer:
    """Improved trainer to fix model bias and accuracy issues"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_to_id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.id_to_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def create_balanced_dataset(self):
        """Create a balanced dataset for training"""
        print("📊 Creating Balanced Training Dataset")
        print("=" * 50)
        
        # Load existing data
        datasets_to_try = ['dataset_combined.csv', 'dataset_fixed.csv', 'dataset.csv']
        df = None
        
        for dataset_file in datasets_to_try:
            if os.path.exists(dataset_file):
                try:
                    df = pd.read_csv(dataset_file)
                    print(f"✅ Loaded {dataset_file}: {len(df)} samples")
                    break
                except Exception as e:
                    print(f"❌ Error loading {dataset_file}: {e}")
                    continue
        
        if df is None:
            print("❌ No valid dataset found. Creating synthetic dataset...")
            df = self.create_synthetic_dataset()
        
        # Fix column names if needed
        if 'extracted_text' in df.columns and 'text' not in df.columns:
            df['text'] = df['extracted_text']
        if 'label' in df.columns and 'sentiment' not in df.columns:
            df['sentiment'] = df['label'].map({'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'})
        
        # Clean data
        df = df.dropna(subset=['text', 'sentiment'])
        df = df[df['text'].str.strip() != '']
        df = df[df['text'] != 'No Text']
        
        # Add more synthetic data to balance classes
        synthetic_data = self.generate_additional_data()
        df_synthetic = pd.DataFrame(synthetic_data)
        df = pd.concat([df, df_synthetic], ignore_index=True)
        
        # Check class distribution
        print(f"\n📈 Class Distribution:")
        class_counts = df['sentiment'].value_counts()
        for sentiment, count in class_counts.items():
            print(f"  {sentiment}: {count} samples")
        
        # Balance classes by upsampling minority classes
        df_balanced = self.balance_classes(df)
        
        print(f"\n📈 Balanced Class Distribution:")
        balanced_counts = df_balanced['sentiment'].value_counts()
        for sentiment, count in balanced_counts.items():
            print(f"  {sentiment}: {count} samples")
        
        return df_balanced
    
    def create_synthetic_dataset(self):
        """Create synthetic dataset if no valid dataset found"""
        synthetic_data = self.generate_additional_data()
        return pd.DataFrame(synthetic_data)
    
    def generate_additional_data(self):
        """Generate additional training data to balance classes"""
        additional_data = []
        
        # Positive examples
        positive_examples = [
            "This meme is absolutely hilarious! 😂😂😂",
            "Best meme ever! Love it so much!",
            "This made my day! So wholesome ❤️",
            "Amazing content! Keep it up!",
            "LOL this is so funny! Can't stop laughing",
            "Wholesome and heartwarming meme",
            "This meme brings me joy",
            "Perfect meme for today! Love it!",
            "This is comedy gold! 😂",
            "Such a clever and funny meme",
            "Absolutely brilliant! 🔥",
            "This is pure gold! Amazing!",
            "So good! Made me smile 😊",
            "Love this so much! Perfect timing",
            "Incredible meme! Well done",
            "This is fantastic! 💯",
            "Awesome content! Love it",
            "This made me laugh out loud!",
            "Such a great meme! Brilliant",
            "Perfect! This is amazing ⭐",
            "Wonderful meme! Love the humor",
            "Excellent work! So funny",
            "This is beautiful! Love it ❤️",
            "Great job! This is hilarious",
            "Outstanding meme! Well done 👏"
        ]
        
        # Negative examples
        negative_examples = [
            "I hate everything about this",
            "This is the worst thing ever",
            "Ugh, another boring Monday",
            "This meme is terrible and unfunny",
            "I'm so tired of this nonsense",
            "This makes me angry and frustrated",
            "What a horrible day",
            "This content is disappointing",
            "I can't stand this anymore",
            "This is so annoying and stupid",
            "Terrible meme! Not funny at all",
            "This is awful! Hate it 😠",
            "Worst content ever! Disgusting",
            "This sucks so much! Bad",
            "Horrible! This is trash",
            "Pathetic meme! Not good",
            "This is garbage! Terrible",
            "Awful content! Disappointing",
            "This is disgusting! Hate it",
            "Terrible! This is awful 👎",
            "Bad meme! Not funny",
            "This is annoying! Stop it",
            "Horrible! This is terrible",
            "Awful! This is bad content",
            "This is frustrating! Hate it 😡"
        ]
        
        # Neutral examples
        neutral_examples = [
            "It's Wednesday my dudes",
            "Today is Thursday",
            "The weather is cloudy",
            "I went to the store",
            "This is a meme",
            "Some text about things",
            "Random meme content here",
            "It is what it is",
            "Things happen sometimes",
            "This exists I guess",
            "Here is some content",
            "This is just a regular post",
            "Normal meme content",
            "Just another day",
            "This is happening now",
            "Some random thoughts",
            "Regular content here",
            "This is a thing",
            "Just posting this",
            "Here we go again",
            "This is ordinary content",
            "Just a regular meme",
            "Standard post here",
            "This is normal",
            "Regular day stuff 🤔"
        ]
        
        # Add to dataset
        for text in positive_examples:
            additional_data.append({'text': text, 'sentiment': 'Positive'})
        
        for text in negative_examples:
            additional_data.append({'text': text, 'sentiment': 'Negative'})
        
        for text in neutral_examples:
            additional_data.append({'text': text, 'sentiment': 'Neutral'})
        
        return additional_data
    
    def balance_classes(self, df):
        """Balance classes by upsampling minority classes"""
        # Find the maximum class size
        class_counts = df['sentiment'].value_counts()
        max_size = class_counts.max()
        
        # Upsample each class to match the maximum size
        balanced_dfs = []
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            class_df = df[df['sentiment'] == sentiment]
            if len(class_df) < max_size:
                # Upsample by repeating samples
                upsampled = class_df.sample(n=max_size, replace=True, random_state=42)
                balanced_dfs.append(upsampled)
            else:
                balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("\n🔧 Preparing Data for Training")
        print("=" * 50)
        
        # Convert labels to numeric
        df['label_id'] = df['sentiment'].map(self.label_to_id)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].values,
            df['label_id'].values,
            test_size=0.2,
            stratify=df['label_id'].values,
            random_state=42
        )
        
        print(f"📊 Data Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        
        # Check class distribution in splits
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        
        print(f"\n📈 Training Set Distribution:")
        for label_id, count in train_dist.items():
            label_name = self.id_to_label[label_id]
            print(f"  {label_name}: {count} samples")
        
        print(f"\n📈 Test Set Distribution:")
        for label_id, count in test_dist.items():
            label_name = self.id_to_label[label_id]
            print(f"  {label_name}: {count} samples")
        
        return X_train, X_test, y_train, y_test
    
    def compute_class_weights(self, y_train):
        """Compute class weights to handle imbalance"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"\n⚖️ Computed Class Weights:")
        for label_id, weight in class_weight_dict.items():
            label_name = self.id_to_label[label_id]
            print(f"  {label_name}: {weight:.3f}")
        
        return class_weight_dict
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the improved model"""
        print("\n🚀 Training Improved Model")
        print("=" * 50)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            label2id=self.label_to_id,
            id2label=self.id_to_label
        )
        
        # Create datasets
        train_dataset = MemeDataset(X_train, y_train, self.tokenizer)
        test_dataset = MemeDataset(X_test, y_test, self.tokenizer)
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Training arguments with improved settings
        training_args = TrainingArguments(
            output_dir='./improved_model',
            num_train_epochs=5,  # More epochs
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./improved_model/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            learning_rate=2e-5,  # Lower learning rate
            gradient_accumulation_steps=2,  # Better gradient updates
            dataloader_drop_last=False,
            seed=42
        )
        
        # Custom trainer with class weights
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # Apply class weights
                weight_tensor = torch.tensor([class_weights[i] for i in range(3)], 
                                           dtype=torch.float32, device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
                loss = loss_fct(logits.view(-1, 3), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        print("🏃‍♂️ Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model('./improved_model_final')
        self.tokenizer.save_pretrained('./improved_model_final')
        
        print("✅ Training completed!")
        print("📁 Model saved to: ./improved_model_final")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
        }
    
    def evaluate_model(self, trainer, X_test, y_test):
        """Evaluate the trained model"""
        print("\n📊 Evaluating Improved Model")
        print("=" * 50)
        
        # Create test dataset
        test_dataset = MemeDataset(X_test, y_test, self.tokenizer)
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📈 Model Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        report = classification_report(y_test, y_pred, 
                                     target_names=['Negative', 'Neutral', 'Positive'])
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, ['Negative', 'Neutral', 'Positive'])
        
        # Test with sample inputs
        self.test_sample_predictions()
        
        return accuracy
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Improved Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('improved_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Confusion matrix saved as: improved_model_confusion_matrix.png")
    
    def test_sample_predictions(self):
        """Test model with sample predictions"""
        print(f"\n🧪 Testing Sample Predictions:")
        print("=" * 50)
        
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
            ("Love this wholesome content ❤️", "Positive")
        ]
        
        correct = 0
        for text, expected in test_cases:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Predict
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
            
            print(f"{status} '{text[:40]}...'")
            print(f"   Predicted: {predicted} (Expected: {expected}) - Confidence: {confidence:.2%}")
        
        accuracy = correct / len(test_cases)
        print(f"\n📊 Sample Test Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
        
        return accuracy

def main():
    """Main training function"""
    print("🔧 FIXING YOUR MEME SENTIMENT MODEL")
    print("=" * 60)
    print("This will address the bias and accuracy issues in your current model")
    print()
    
    # Initialize trainer
    trainer = ImprovedModelTrainer()
    
    # Create balanced dataset
    df = trainer.create_balanced_dataset()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train improved model
    model_trainer = trainer.train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    accuracy = trainer.evaluate_model(model_trainer, X_test, y_test)
    
    print(f"\n🎯 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"✅ Improved model accuracy: {accuracy:.2%}")
    print(f"📁 Model saved to: ./improved_model_final")
    print(f"📊 Confusion matrix: improved_model_confusion_matrix.png")
    
    print(f"\n📋 Next Steps:")
    print("1. Test the improved model: python test_improved_model.py")
    print("2. Compare with old model performance")
    print("3. Deploy the improved model")
    
    return accuracy

if __name__ == "__main__":
    main()
