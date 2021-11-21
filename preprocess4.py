import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import json
import glob
import re
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# for visualize
#import pyLDAvis
#import pyLDAvis.gensim



df = pd.read_csv('2017-11.csv')
texts = df['body']

# Clean the reddit comments
def clean_reddit(text):

    text = str(text)
    # newline
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    text = re.sub(r'\s\s+', ' ', text)

    # Quotes
    text = re.sub(r'\"?\\?&?gt;?', '', text)

    # Bullet points/asterisk (bold/italic)
    
    text = re.sub(r'\*', '', text)
    text = re.sub('&amp;#x200B;', '', text)

    # []() Link (Also removes the hyperlink)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # Strikethrough
   
    text = re.sub('~', '', text)

    # Spoiler, which is used with < less-than (Preserves the text)
    
    text = re.sub('&lt;', '', text)
    text = re.sub(r'!(.*?)!', r'\1', text)
    # Code, inline and block
    
    text = re.sub('`', '', text)

    # Superscript (Preserves the text)
    
    text = re.sub(r'\^\((.*?)\)', r'\1', text)

    # Table
    
    text = re.sub(r'\|', ' ', text)
    text = re.sub(':-', '', text)

    # Heading
    
    text = re.sub('#', '', text)

    return text

# Clean NLP texts

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def tokenize(text):
    return nltk.tokenize.word_tokenize(text)

def remove_stopwords(text):
    text=[word for word in text if word not in stop_words]
    return text

def lemmatize(text):
    wn = nltk.WordNetLemmatizer()
    w = [wn.lemmatize(word) for word in text]
    return w
nlp = spacy.load("en_core_web_sm", disable = ['parser', 'ner'])
def pos_pick(text, allowed_postags = ["NOUN", "ADJ", 'VERB', 'ADV']):
    text = nlp(text)
    new_token = []
    for token in text:
        if token.pos_ in allowed_postags:
            new_token.append(token.lemma_)
    return new_token


def clean_nlp(text):
    text = clean_reddit(text)
    text = remove_punctuation(text)
    tokens = pos_pick(text)
    tokens = remove_stopwords(tokens)
    #tokens = lemmatize(tokens)
    return tokens

def load_data4():
    clean_ar = [clean_nlp(text) for text in texts]
    return clean_ar

df = pd.DataFrame()
df['old_text'] = texts
df['clean_text'] = load_data4()

df.to_csv('clean_nlp_text.csv', index=False)


