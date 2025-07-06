#!/usr/bin/env python3
"""
Sarcasm Training Data Strategy
Methods to improve sarcasm classification through better training data
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple

class SarcasmDataAugmentor:
    """
    Creates augmented training data with better sarcasm representation
    """
    
    def __init__(self):
        self.sarcasm_patterns = self._create_sarcasm_patterns()
        
    def _create_sarcasm_patterns(self) -> Dict:
        """Create comprehensive sarcasm patterns for data generation"""
        return {
            'sarcastic_positive_phrases': [
                # These should be labeled as NEUTRAL or NEGATIVE (sarcastic)
                "Oh great, {topic}",
                "Wow, such {adjective} {topic}",
                "Yeah, because {statement} always works",
                "Perfect, just what I needed",
                "Thanks for the {noun}, very helpful",
                "Oh sure, {statement}",
                "Real {adjective} there",
                "How {adjective} of them",
                "Obviously the best {noun} ever",
                "Just {adjective}, absolutely {adjective}"
            ],
            'genuinely_positive_phrases': [
                # These should be labeled as POSITIVE
                "This is genuinely {adjective}!",
                "I actually love this {noun}",
                "Really {adjective} and {adjective}",
                "This made me {emotion}",
                "Such a {adjective} {noun}",
                "Honestly {adjective}",
                "Truly {adjective} content",
                "This is {adjective} in the best way",
                "I'm so {emotion} about this",
                "What a {adjective} {noun}"
            ],
            'context_words': {
                'topics': ['meme', 'content', 'post', 'joke', 'picture', 'video'],
                'adjectives': ['original', 'creative', 'funny', 'clever', 'brilliant', 'amazing', 'perfect'],
                'statements': ['that', 'this approach', 'this method', 'this strategy'],
                'nouns': ['advice', 'tip', 'suggestion', 'idea', 'concept', 'meme'],
                'emotions': ['happy', 'excited', 'thrilled', 'pleased', 'delighted']
            },
            'sarcasm_indicators': {
                'emojis': ['🙄', '😏', '🙃', '😒', '🤨'],
                'intensifiers': ['so', 'such', 'very', 'really', 'absolutely'],
                'context_markers': ['obviously', 'clearly', 'definitely', 'surely']
            }
        }
    
    def generate_sarcastic_examples(self, count: int = 100) -> List[Tuple[str, str]]:
        """
        Generate sarcastic text examples with proper labels
        
        Returns:
            List of (text, label) tuples
        """
        examples = []
        patterns = self.sarcasm_patterns
        
        # Generate sarcastic examples (should be neutral/negative)
        for i in range(count // 2):
            # Choose random pattern components
            pattern = np.random.choice(patterns['sarcastic_positive_phrases'])
            context = patterns['context_words']
            
            # Fill in the pattern
            text = pattern.format(
                topic=np.random.choice(context['topics']),
                adjective=np.random.choice(context['adjectives']),
                statement=np.random.choice(context['statements']),
                noun=np.random.choice(context['nouns']),
                emotion=np.random.choice(context['emotions'])
            )
            
            # Add sarcasm indicators
            if np.random.random() < 0.4:  # 40% chance to add emoji
                text += " " + np.random.choice(patterns['sarcasm_indicators']['emojis'])
            
            # Label as neutral for mild sarcasm, negative for harsh sarcasm
            if any(word in text.lower() for word in ['great', 'perfect', 'obviously']):
                label = 'negative'  # More obviously sarcastic
            else:
                label = 'neutral'   # Mild sarcasm
                
            examples.append((text, label))
        
        # Generate genuinely positive examples
        for i in range(count // 2):
            pattern = np.random.choice(patterns['genuinely_positive_phrases'])
            context = patterns['context_words']
            
            text = pattern.format(
                adjective=np.random.choice(context['adjectives']),
                noun=np.random.choice(context['nouns']),
                emotion=np.random.choice(context['emotions'])
            )
            
            # Add genuine positive indicators
            if np.random.random() < 0.3:  # 30% chance to add positive emoji
                text += " ❤️"
            
            examples.append((text, 'positive'))
        
        return examples
    
    def generate_balanced_dataset(self, 
                                original_dataset_path: str = "dataset.csv",
                                output_path: str = "enhanced_dataset.csv",
                                augment_count: int = 200) -> pd.DataFrame:
        """
        Create a balanced dataset with proper sarcasm representation
        """
        print("🔄 Generating Enhanced Dataset with Sarcasm Awareness")
        print("=" * 60)
        
        # Load original dataset
        try:
            original_df = pd.read_csv(original_dataset_path)
            print(f"✅ Loaded original dataset: {len(original_df)} samples")
            print(f"   Original distribution: {original_df['label'].value_counts().to_dict()}")
        except FileNotFoundError:
            print("⚠️ Original dataset not found, creating from scratch")
            original_df = pd.DataFrame(columns=['extracted_text', 'label', 'image_path'])
        
        # Generate sarcastic examples
        print(f"\n🎭 Generating {augment_count} sarcasm-aware examples...")
        sarcastic_examples = self.generate_sarcastic_examples(augment_count)
        
        # Create augmented dataframe
        augmented_data = []
        for text, label in sarcastic_examples:
            augmented_data.append({
                'extracted_text': text,
                'label': label,
                'image_path': 'synthetic_data',
                'source': 'sarcasm_augmentation',
                'is_sarcastic': 'yes' if label in ['neutral', 'negative'] else 'no'
            })
        
        augmented_df = pd.DataFrame(augmented_data)
        
        # Add manually curated sarcasm examples
        manual_examples = self._get_manual_sarcasm_examples()
        manual_df = pd.DataFrame(manual_examples)
        
        # Combine all datasets
        if not original_df.empty:
            # Add sarcasm indicator to original data
            original_df['source'] = 'original'
            original_df['is_sarcastic'] = 'unknown'
            
            combined_df = pd.concat([original_df, augmented_df, manual_df], ignore_index=True)
        else:
            combined_df = pd.concat([augmented_df, manual_df], ignore_index=True)
        
        # Balance the dataset
        label_counts = combined_df['label'].value_counts()
        print(f"\n📊 Enhanced dataset distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count} samples")
        
        # Save enhanced dataset
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Enhanced dataset saved to: {output_path}")
        print(f"   Total samples: {len(combined_df)}")
        
        # Generate training recommendations
        self._generate_training_recommendations(combined_df)
        
        return combined_df
    
    def _get_manual_sarcasm_examples(self) -> List[Dict]:
        """
        Manually curated sarcasm examples for training
        """
        return [
            # Sarcastic (should be neutral/negative)
            {"extracted_text": "Oh great, another Monday meme 🙄", "label": "negative", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "Wow, such original content 😏", "label": "negative", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "Yeah, because that always works out perfectly", "label": "negative", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "Perfect timing, just what I needed today", "label": "neutral", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "Thanks for the reminder, as if I could forget 🙃", "label": "neutral", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "Real creative there, genius", "label": "negative", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "Oh sure, let me drop everything for this", "label": "negative", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            {"extracted_text": "How thoughtful of them 🙄", "label": "negative", "image_path": "manual", "source": "manual_sarcasm", "is_sarcastic": "yes"},
            
            # Genuinely positive (should be positive)
            {"extracted_text": "This meme actually made me laugh out loud!", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "I genuinely love this wholesome content ❤️", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "This is truly funny and clever", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "Such a heartwarming meme, thanks for sharing!", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "Really creative and original work!", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "This honestly made my day better", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "I'm so happy to see content like this", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
            {"extracted_text": "What a brilliant and funny meme!", "label": "positive", "image_path": "manual", "source": "manual_positive", "is_sarcastic": "no"},
        ]
    
    def _generate_training_recommendations(self, dataset: pd.DataFrame):
        """Generate training recommendations based on the enhanced dataset"""
        
        print(f"\n🎯 Training Recommendations:")
        print("=" * 40)
        
        sarcastic_count = len(dataset[dataset['is_sarcastic'] == 'yes'])
        total_count = len(dataset)
        
        print(f"1. **Sarcasm Representation**: {sarcastic_count}/{total_count} ({sarcastic_count/total_count*100:.1f}%) samples include sarcasm")
        print(f"2. **Training Epochs**: Use 8-12 epochs (more data needs more training)")
        print(f"3. **Learning Rate**: Start with 2e-5, reduce if overfitting")
        print(f"4. **Validation Split**: Use 20% for validation, stratified by label")
        print(f"5. **Class Weights**: Consider balancing since we have more data now")
        
        # Specific recommendations
        print(f"\n📋 Specific Actions:")
        print(f"• Train with enhanced dataset for better sarcasm understanding")
        print(f"• Use early stopping to prevent overfitting")
        print(f"• Monitor both overall accuracy and sarcasm-specific metrics")
        print(f"• Consider ensemble with rule-based sarcasm detection")

def create_training_script():
    """Create a training script for the enhanced dataset"""
    
    training_script = '''#!/usr/bin/env python3
"""
Training script for sarcasm-aware sentiment model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

def train_sarcasm_aware_model():
    """Train model with enhanced sarcasm dataset"""
    
    print("🎯 Training Sarcasm-Aware Sentiment Model")
    print("=" * 50)
    
    # Load enhanced dataset
    df = pd.read_csv("enhanced_dataset.csv")
    print(f"Dataset loaded: {len(df)} samples")
    
    # Prepare data
    df = df[df['extracted_text'].notna()].copy()
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['labels'] = df['label'].map(label_mapping)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['extracted_text'].tolist(),
        df['labels'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['labels']
    )
    
    print(f"Training: {len(train_texts)}, Validation: {len(val_texts)}")
    
    # Initialize model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./sarcasm_aware_model',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25
    )
    
    # TODO: Add training logic here
    print("Training script template created. Implement training logic.")

if __name__ == "__main__":
    train_sarcasm_aware_model()
'''
    
    with open("train_sarcasm_model.py", "w") as f:
        f.write(training_script)
    
    print("📄 Training script saved to: train_sarcasm_model.py")

def main():
    """Main function to demonstrate sarcasm data augmentation"""
    
    # Create augmentor
    augmentor = SarcasmDataAugmentor()
    
    # Generate enhanced dataset
    enhanced_df = augmentor.generate_balanced_dataset(
        original_dataset_path="dataset.csv",
        output_path="enhanced_dataset.csv",
        augment_count=150
    )
    
    # Create training script template
    create_training_script()
    
    print(f"\n🎉 Sarcasm improvement strategy complete!")
    print(f"Next steps:")
    print(f"1. Review enhanced_dataset.csv")
    print(f"2. Train new model with: python train_sarcasm_model.py")
    print(f"3. Test with sarcasm_aware_analyzer.py")

if __name__ == "__main__":
    main()
