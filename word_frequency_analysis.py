import re
from collections import Counter
import sys
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

def load_text(file_path_or_url):
    if file_path_or_url == 'urllist':
        combined_text = ''
        with open('urllist', 'r') as url_file:
            for url in url_file:
                url = url.strip()
                if url.startswith('http://') or url.startswith('https://'):
                    response = requests.get(url)
                    response.raise_for_status()
                    combined_text += response.text + '\n'
        return combined_text
    elif file_path_or_url.startswith('http://') or file_path_or_url.startswith('https://'):
        response = requests.get(file_path_or_url)
        response.raise_for_status()
        return response.text
    else:
        with open(file_path_or_url, 'r', encoding='utf-8') as file:
            return file.read()

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token.isalpha() and token not in stop_words
    ]
    
    return ' '.join(cleaned_tokens)

def get_word_frequencies(text):
    words = text.split()
    return Counter(words)

def generate_wordcloud(word_frequencies):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def main(file_path):
    text = load_text(file_path)
    cleaned_text = clean_text(text)
    word_frequencies = get_word_frequencies(cleaned_text)
    
    for word, freq in word_frequencies.most_common():
        print(f'{word}: {freq}')

    generate_wordcloud(word_frequencies)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python word_frequency_analysis.py <file_path>")
    else:
        main(sys.argv[1])