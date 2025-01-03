import re
import logging
from collections import Counter
import sys
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_text(file_path_or_url):
    """
    Load text from a file or URL.
    If 'urllist' is provided, load text from multiple URLs listed in the file.
    """
    logging.info(f'Loading text from {file_path_or_url}')
    if file_path_or_url == 'urllist':
        combined_text = ''
        with open('./source/urllist', 'r') as url_file:
            for url in url_file:
                url = url.strip()
                if url.startswith('http://') or url.startswith('https://'):
                    logging.info(f'Fetching text from URL: {url}')
                    response = requests.get(url)
                    response.raise_for_status()
                    combined_text += response.text + '\n'
        return combined_text
    elif file_path_or_url.startswith('http://') or file_path_or_url.startswith('https://'):
        logging.info(f'Fetching text from URL: {file_path_or_url}')
        response = requests.get(file_path_or_url)
        response.raise_for_status()
        return response.text
    else:
        logging.info(f'Reading text from file: {file_path_or_url}')
        with open(file_path_or_url, 'r', encoding='utf-8') as file:
            return file.read()

def save_text_to_file(text, filename):
    """
    Save the given text to a file.
    """
    logging.info(f'Saving text to file: {filename}')
    with open(os.path.join('./export', filename), 'w', encoding='utf-8') as file:
        file.write(text)

def clean_text(text):
    """
    Clean the input text by removing HTML tags, converting to lowercase,
    tokenizing, removing stopwords, and lemmatizing.
    """
    logging.info('Cleaning text')
    # Parse HTML content and extract text
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
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
    """
    Get word frequencies from the cleaned text.
    Only count words that appear more than once and return the top 40 words.
    Exclude words from 'wordfilter' file from the count.
    """
    logging.info('Calculating word frequencies')
    
    # Load words to exclude from 'wordfilter' file
    with open('wordfilter', 'r', encoding='utf-8') as file:
        excluded_words = set(file.read().splitlines())
    
    words = text.split()
    word_counts = Counter(words)
    filtered_counts = Counter({word: count for word, count in word_counts.items() if count > 1 and word not in excluded_words})
    return Counter(dict(filtered_counts.most_common(50)))

def calculate_tfidf(text):
    """
    Calculate TF-IDF scores for the given text, excluding certain words.
    """
    logging.info('Calculating TF-IDF scores')
    
    # Load words to exclude from 'wordfilter' file
    with open('wordfilter', 'r', encoding='utf-8') as file:
        excluded_words = set(file.read().splitlines())
    
    # Tokenize text and filter out excluded words
    tokens = text.split()
    filtered_text = ' '.join([word for word in tokens if word not in excluded_words])
    
    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([filtered_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    return Counter(tfidf_scores)

def generate_wordcloud(word_frequencies):
    """
    Generate and display a word cloud from word frequencies.
    """
    logging.info('Generating word cloud')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def save_wordcloud(word_frequencies, filename="exportedwordcloud.png"):
    """
    Save the word cloud to a file.
    """
    logging.info(f'Saving word cloud to {filename}')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    wordcloud.to_file(os.path.join('./export', filename))

def generate_plotly_bar_chart(word_frequencies):
    """
    Generate and display a bar chart from word frequencies using Plotly.
    """
    logging.info('Generating bar chart with Plotly')
    # Convert the Counter object to a dictionary
    word_freq_dict = dict(word_frequencies)
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])
    # Sort the DataFrame by frequency
    df = df.sort_values(by='Frequency', ascending=False)
    # Generate the bar chart
    fig = px.bar(df, x='Word', y='Frequency', title='Word Frequencies')
    fig.show()

def save_plotly_bar_chart(word_frequencies, filename="exportedbarchart.html", image_filename="exportedbarchart.png"):
    """
    Save the Plotly bar chart to a file and as an image.
    """
    logging.info(f'Saving Plotly bar chart to {filename} and {image_filename}')
    # Convert the Counter object to a dictionary
    word_freq_dict = dict(word_frequencies)
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])
    # Sort the DataFrame by frequency
    df = df.sort_values(by='Frequency', ascending=False)
    # Generate the bar chart
    fig = px.bar(df, x='Word', y='Frequency', title='Word Frequencies')
    # Save the bar chart to a file
    fig.write_html(os.path.join('./export', filename))
    # Save the bar chart as an image
    fig.write_image(os.path.join('./export', image_filename))

def save_word_frequencies_score_to_file(word_frequencies, tfidf_scores, filename="word_frequencies.csv"):
    """
    Save the word frequencies and TF-IDF scores to a file in table form (CSV format).
    """
    logging.info(f'Saving word frequencies and TF-IDF scores to file: {filename}')
    # Convert the Counter object to a dictionary
    word_freq_dict = dict(word_frequencies)
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])
    # Add TF-IDF scores to the DataFrame
    df['TF-IDF'] = df['Word'].apply(lambda word: tfidf_scores.get(word, 0))
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join('./export', filename), index=False)

def main(file_path):
    """
    Main function to load text, clean it, get word frequencies, and generate a word cloud.
    """
    logging.info('Starting main function')
    text = load_text(file_path)
    save_text_to_file(text, 'lastloadedtext')
    cleaned_text = clean_text(text)
    word_frequencies = get_word_frequencies(cleaned_text)
    tfidf_scores = calculate_tfidf(cleaned_text)
    
    print("Word Frequencies and TF-IDF Scores:")
    for word, freq in word_frequencies.most_common():
        score = tfidf_scores.get(word, 0)
        print(f'{word}: {freq}, {score:.4f}')
    
    generate_wordcloud(word_frequencies)
    generate_plotly_bar_chart(word_frequencies)
    save_wordcloud(word_frequencies)
    save_plotly_bar_chart(word_frequencies)
    save_word_frequencies_score_to_file(word_frequencies,tfidf_scores)

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        logging.error("Usage: python word_frequency_analysis.py <file_path>")
        sys.exit("Usage: python word_frequency_analysis.py <file_path>")
    else:
        main(sys.argv[1])