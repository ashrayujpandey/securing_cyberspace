import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
# punkt: For tokenization
# stopwords: For removing common words
# wordnet: For lemmatization
# averaged_perceptron_tagger: For part-of-speech tagging
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def clean_text(text):
    """
    Enhanced text cleaning function that performs multiple cleaning steps.
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: Cleaned text with:
            - Converted to lowercase
            - Removed URLs
            - Removed special characters and numbers
            - Normalized whitespace
    """
    if isinstance(text, str):
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove URLs using regex pattern
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers, replacing them with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace and normalize to single spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

def remove_stopwords(text):
    """
    Remove common stopwords and custom stopwords from the text.
    
    Args:
        text (str): The input text
        
    Returns:
        str: Text with stopwords removed
    """
    # Get standard English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords specific to our domain
    # These are common words in entertainment content that might cause false positives
    custom_stopwords = {
        'movie', 'film', 'book', 'show', 'series', 
        'documentary', 'novel', 'story', 'scene', 'plot'
    }
    stop_words.update(custom_stopwords)
    
    # Tokenize the text and remove stopwords
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)

def lemmatize_text(text):
    """
    Convert words to their base form using lemmatization.
    This helps reduce words to their dictionary form (e.g., 'running' -> 'run')
    
    Args:
        text (str): The input text
        
    Returns:
        str: Text with words lemmatized
    """
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized_text)

def preprocess_data(text):
    """
    Main preprocessing pipeline that combines all cleaning steps.
    
    Args:
        text (str): The input text to process
        
    Returns:
        str: Fully processed text ready for model input
    """
    # Step 1: Basic cleaning (lowercase, remove special chars, etc.)
    cleaned_text = clean_text(text)
    
    # Step 2: Remove stopwords
    text_no_stopwords = remove_stopwords(cleaned_text)
    
    # Step 3: Lemmatize words to their base form
    processed_text = lemmatize_text(text_no_stopwords)
    
    return processed_text

if __name__ == "__main__":
    # Test the preprocessing pipeline with a sample text
    sample_text = "I love this movie! The action scenes were so intense and the story kept me hooked till the end."
    processed = preprocess_data(sample_text)
    print("Original:", sample_text)
    print("Processed:", processed)
