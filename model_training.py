# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os
from data_preparation import preprocess_data
import re

class TerrorismDetectionModel:
    """
    A class for training and using a text classification model to detect suspicious content.
    
    This class handles:
    - Text vectorization using TF-IDF
    - Model training
    - Making predictions
    - Saving and loading the model
    """
    
    def __init__(self):
        """
        Initialize the model with:
        - TF-IDF vectorizer with n-gram features (1-3 words)
        - Logistic Regression classifier with balanced class weights
        """
        # Improve vectorizer settings
        self.vectorizer = TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Adjust model to be more sensitive to suspicious content
        self.model = LogisticRegression(
            random_state=42, 
            class_weight={0: 1, 1: 2},
            max_iter=1000
        )

        # Define threat keywords with weights
        self.threat_keywords = {
            'high_risk': [
                'attack', 'bomb', 'explosive', 'jihad', 'martyr', 'kill', 'murder',
                'execute', 'assassinate', 'hijack', 'detonate', 'gunfire', 'suicide',
                'terrorist', 'massacre', 'behead', 'firebomb', 'sabotage', 'annihilate'
            ],
            'medium_risk': [
                'radical', 'extremist', 'caliphate', 'militant', 'infidel', 'hostage',
                'violence', 'war', 'strike', 'retaliate', 'ambush', 'sniper', 'target',
                'threat', 'hate', 'purge', 'rebel', 'recruit', 'holy war', 'weapons',
                'bloodshed', 'radicalize', 'execution', 'uprising'
            ]
        }

    def create_sample_data(self):
        """Create a balanced dataset with diverse examples."""
        texts = [
            # Positive/Normal content (0)
            "I really appreciate how this promotes unity and mutual respect.",
            "The weather is beautiful today and perfect for a picnic.",
            "I love spending time with my family and friends.",
            "This restaurant serves the most delicious food.",
            "The concert last night was amazing!",
            "Education is the key to a better future.",
            "Let's work together to build a better community.",
            "Celebrating diversity makes our society stronger.",
            "The children had so much fun at the park today.",
            "Working hard to achieve our dreams is important.",
            
            # Safe content with potentially triggering words (0)
            "The movie showed how peace prevailed over violence.",
            "History teaches us about past conflicts and their resolution.",
            "The documentary explored ways to prevent extremism.",
            "In the game, players defend their base from attacks.",
            "The book discusses the importance of conflict resolution.",
            "News report: Peace talks successful in reducing tensions.",
            "Research shows that cooperation beats conflict.",
            "The peace rally promoted non-violent solutions.",
            "Students learn about historical wars and peace treaties.",
            "The art exhibition promotes cultural understanding.",
            
            # Suspicious content (1)
            "We must attack them immediately and show no mercy.",
            "The bomb is ready for detonation at the location.",
            "Our militant group will strike fear into their hearts.",
            "We will execute our plan to eliminate all opposition.",
            "Join our holy war against the infidels.",
            "The weapons are prepared for our upcoming operation.",
            "We must purge our land of these people.",
            "Our suicide mission will bring glory to our cause.",
            "The time has come to behead our enemies.",
            "We will sabotage their infrastructure tonight."
        ]
        
        # First 20 are normal (0), last 10 are suspicious (1)
        labels = [0]*20 + [1]*10
        return texts, labels

    def analyze_text_content(self, text):
        """
        Analyze text for threat keywords while considering context.
        Returns a threat score and relevant keywords found.
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Initialize threat metrics
        threat_score = 0
        found_keywords = []
        
        # Check for high-risk keywords (higher weight)
        for keyword in self.threat_keywords['high_risk']:
            if keyword in text_lower:
                threat_score += 3  # Increased weight for high-risk words
                found_keywords.append(keyword)
        
        # Check for medium-risk keywords
        for keyword in self.threat_keywords['medium_risk']:
            if keyword in text_lower:
                threat_score += 1
                found_keywords.append(keyword)
        
        # Context modifiers
        safe_contexts = [
            'movie', 'game', 'book', 'story', 'fiction', 'film', 'documentary',
            'history', 'historical', 'research', 'study', 'analysis', 'report',
            'news', 'article', 'review', 'discuss', 'examine', 'explore'
        ]
        
        # Reduce threat score for safe contexts
        for context in safe_contexts:
            if context in text_lower:
                threat_score = max(0, threat_score - 2)  # Stronger reduction for safe contexts
        
        return threat_score, found_keywords

    def train(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (list): List of text samples
            y (list): List of corresponding labels (0 for normal, 1 for suspicious)
        """
        # Preprocess all text samples
        X_processed = [preprocess_data(text) for text in X]
        
        # Convert text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X_processed)
        
        # Train the model
        self.model.fit(X_tfidf, y)

    def predict(self, text):
        """
        Make a prediction for a single text sample.
        
        Args:
            text (str): The text to classify
            
        Returns:
            tuple: (prediction, probability, found_keywords, threat_score)
                - prediction: 0 for normal, 1 for suspicious
                - probability: Confidence score for the prediction
                - found_keywords: List of keywords found in the text
                - threat_score: Threat score based on the text content
        """
        # Preprocess the input text
        processed_text = preprocess_data(text)
        
        # Convert to TF-IDF features
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Get base prediction from the model
        base_prediction = self.model.predict(text_tfidf)[0]
        base_probability = self.model.predict_proba(text_tfidf)[0][1]
        
        # Get threat analysis
        threat_score, found_keywords = self.analyze_text_content(text)
        
        # Adjust probability based on threat analysis
        adjusted_probability = base_probability
        if threat_score > 0:
            # More aggressive adjustment for high threat scores
            adjusted_probability = min(1.0, base_probability + (threat_score * 0.2))
        
        # Final prediction based on adjusted probability
        final_prediction = 1 if adjusted_probability > 0.5 else 0
        
        return final_prediction, adjusted_probability, found_keywords, threat_score

    def save_model(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """
        Save the trained model and vectorizer to disk.
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model and vectorizer
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

def create_sample_data():
    """
    Create a balanced dataset with diverse examples.
    
    Returns:
        tuple: (texts, labels)
            - texts: List of text samples
            - labels: List of corresponding labels (0 for normal, 1 for suspicious)
    """
    # Create a diverse and balanced dataset
    texts = [
        # Positive/Normal content (0)
        "I really appreciate how this promotes unity and mutual respect.",
        "The weather is beautiful today and perfect for a picnic.",
        "I love spending time with my family and friends.",
        "This restaurant serves the most delicious food.",
        "The concert last night was amazing!",
        "Education is the key to a better future.",
        "Let's work together to build a better community.",
        "Celebrating diversity makes our society stronger.",
        "The children had so much fun at the park today.",
        "Working hard to achieve our dreams is important.",
        "I love spending time with my family on the weekends."

"This book really changed my perspective on life."

"Great job on your presentation today — very impressive!"

"The weather is perfect for a walk in the park."

"Kindness is the most powerful tool we have."

"She always helps others without expecting anything in return."

"I'm looking forward to the team outing next week!"

"Happy birthday! Wishing you a day filled with joy and laughter."

"This meal tastes amazing, thank you for cooking it!"

"I enjoy learning about different cultures and traditions."

"Let's work together to build a better future."

"Volunteering at the shelter was such a rewarding experience."

"He gave an inspiring speech about peace and tolerance."

"Yoga and meditation have really improved my mental health."

"The community came together to support the local school."

"She donated to the charity without any recognition."

"Listening to music helps me relax after a long day."

"They welcomed us with warmth and hospitality."

"Education is the key to a brighter tomorrow."

"Let's plant more trees and take care of our environment."


        
        # Safe content with potentially triggering words (0)
        "The movie showed how peace prevailed over violence.",
        "History teaches us about past conflicts and their resolution.",
        "The documentary explored ways to prevent extremism.",
        "In the game, players defend their base from attacks.",
        "The book discusses the importance of conflict resolution.",
        "News report: Peace talks successful in reducing tensions.",
        "Research shows that cooperation beats conflict.",
        "The peace rally promoted non-violent solutions.",
        "Students learn about historical wars and peace treaties.",
        "The art exhibition promotes cultural understanding.",
        
        # Suspicious content (1)
        "They deserve to be wiped out completely."

"We need to plan a serious attack on them."

"Eliminate every single one of them without mercy."

"He should be executed for his actions."

"Our enemies must burn for what they've done."

"Spread chaos and make them suffer."

"Time to strike back with full force."

"Recruit more people to join our holy war."

"Death to the infidels who oppose our mission."

"This is the perfect moment for a bombing."

"Martyrdom is the only path to justice."

"Destroy everything they've built."

"We'll hijack their resources and cripple them."

"It's time for bloodshed and revenge."

"Their entire community should be purged."

"We must assassinate their leader to send a message."

"Prepare the explosives and wait for my signal."

"Their lives mean nothing to us."

"Only violence can fix this world."

"Let's radicalize more people into the cause."
        "We must attack them immediately and show no mercy.",
        "The bomb is ready for detonation at the location.",
        "Our militant group will strike fear into their hearts.",
        "We will execute our plan to eliminate all opposition.",
        "Join our holy war against the infidels.",
        "The weapons are prepared for our upcoming operation.",
        "We must purge our land of these people.",
        "Our suicide mission will bring glory to our cause.",
        "The time has come to behead our enemies.",
        "We will sabotage their infrastructure tonight."
    ]
    
    # First 20 are normal (0), last 10 are suspicious (1)
    labels = [0]*20 + [1]*10
    return texts, labels

if __name__ == "__main__":
    # Initialize model
    detector = TerrorismDetectionModel()
    
    # Create and split dataset
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Train model
    detector.train(X_train, y_train)
    
    # Save the model
    detector.save_model()
    
    # Test some examples
    test_texts = [
        "I really appreciate how this promotes unity and mutual respect.",
        "The weather is beautiful today.",
        "We must attack them immediately.",
        "The movie had some violent scenes.",
        "Let's work together for peace.",
    ]
    
    # Process and predict test texts
    test_processed = [preprocess_data(text) for text in test_texts]
    test_features = detector.vectorizer.transform(test_processed)
    predictions = detector.model.predict(test_features)
    probabilities = detector.model.predict_proba(test_features)
    
    print("\nTesting predictions:")
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        print(f"\nText: {text}")
        print(f"Prediction: {'Suspicious' if pred == 1 else 'Normal'}")
        print(f"Probability of being suspicious: {prob[1]:.2f}")
    
    print("\nModel saved successfully!")
