from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from s3client import s3Loader
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import io

load_dotenv()


def load_models_from_s3(s3_client, bucket_name, model_names):
    """Load models from S3."""
    models = {}
    for model_name in model_names:
        model = s3_client.load_model_from_s3(bucket_name, f"{model_name}_model.pkl")
        models[model_name] = model
    return models

def transform_data(tfidf, text_series):
    """Transform text data using the loaded TF-IDF model."""
    return tfidf.transform(text_series)

def extract_topics(nmf_model, tfidf, dtm):
    """Extract topics using the loaded NMF model."""
    H = nmf_model.components_
    words = np.array(tfidf.get_feature_names_out())
    topic_words = pd.DataFrame(columns=[f'Word {i + 1}' for i in range(15)])
    
    for i in range(nmf_model.n_components):
        topic_words.loc[f'Topic {i + 1}'] = words[H[i].argsort()[::-1][:15]]
    
    return topic_words

def assign_topics(nmf_model, dtm):
    """Assign topics to documents."""
    topic_results = nmf_model.transform(dtm)
    return topic_results.argmax(axis=1)

def save_results_to_s3(s3_client, df, bucket_name, object_key):
    """Save the results to S3."""
    p, _ = os.path.splitext(object_key)
    new_object_key = f"preprocessed_labeled/{p.split('/')[1]}.csv"
    s3_client.save_df_to_s3(df, bucket_name, new_object_key)

if __name__ == "__main__":
    s3_client = s3Loader()
    clean_bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")
    object_key = 'preprocessed/complaints-2021-05-14_08_16_.csv'

    data = s3_client.load_csv_from_s3(object_key=object_key, bucket_name=clean_bucket_name)
    df = pd.read_csv(io.BytesIO(data))
    df['complaint_final'] = df['complaint_final'].replace(np.nan, "")

    model_names = ['tfidf', 'nmf']
    models = load_models_from_s3(s3_client, clean_bucket_name, model_names)
    tfidf = models['tfidf']
    nmf_model = models['nmf']

    dtm = transform_data(tfidf, df['complaint_final'])
    topic_words = extract_topics(nmf_model, tfidf, dtm)
    df['Topic'] = assign_topics(nmf_model, dtm)

    save_results_to_s3(s3_client, df, clean_bucket_name, object_key)