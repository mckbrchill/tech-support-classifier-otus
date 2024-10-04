import os
import re
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import spacy
from textblob import TextBlob
from s3client import s3Loader
import nltk

# Load necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load environment variables
load_dotenv()


def clean_text(text):
    """Clean input text by removing punctuation and numeric words."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def lemmatize_text(text):
    """Lemmatize input text using spaCy."""
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)

def get_pos_tags(text):
    """Extract nouns from input text using TextBlob part-of-speech tagging."""
    blob = TextBlob(text)
    return " ".join(word for word, tag in blob.tags if tag == 'NN')

def preprocess_complaints(data):
    """Preprocess the complaints data."""
    df = pd.json_normalize(data)
    df.columns = ['index', 'type', 'id', 'score', 'tags', 'zip_code', 'complaint_id', 'issue', 'date_received',
                  'state', 'consumer_disputed', 'product', 'company_response', 'company', 'submitted_via',
                  'date_sent_to_company', 'company_public_response', 'sub_product', 'timely',
                  'complaint_what_happened', 'sub_issue', 'consumer_consent_provided']

    df['complaint_what_happened'].replace('', np.nan, inplace=True)
    df.dropna(subset=['complaint_what_happened'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['complaint_what_happened'] = df['complaint_what_happened'].astype(str)
    df['complaint_cleaned'] = df['complaint_what_happened'].apply(clean_text)
    df['complaint_lemmatized'] = df['complaint_cleaned'].apply(lemmatize_text)
    df['complaint_pos_filtered'] = df['complaint_lemmatized'].apply(get_pos_tags)
    df['complaint_final'] = df['complaint_pos_filtered'].str.replace('-PRON-', '').str.replace('xxxx', '')

    return df

if __name__ == "__main__":
    s3_client = s3Loader()

    raw_bucket_name = os.environ.get("S3_RAW_BUCKET_NAME")
    clean_bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")
    object_key = 'complaints-2021-05-14_08_16_.json'

    nlp = spacy.load("en_core_web_sm")
    data = s3_client.load_json_from_s3(object_key=object_key, bucket_name=raw_bucket_name)
    df = preprocess_complaints(data)

    p, _ = os.path.splitext(object_key)
    new_object_key = f"preprocessed/{p}.csv"
    s3_client.save_df_to_s3(df, clean_bucket_name, new_object_key)