from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from s3client import s3Loader
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import io

load_dotenv()

def vectorize_text(text_series):
    """Vectorize text using TF-IDF."""
    tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')
    return tfidf, tfidf.fit_transform(text_series)

def extract_topics(dtm, num_topics, tfidf):
    """Extract topics using NMF."""
    nmf_model = NMF(n_components=num_topics, random_state=40)
    nmf_model.fit(dtm)
    H = nmf_model.components_
    words = np.array(tfidf.get_feature_names_out())
    topic_words = pd.DataFrame(columns=[f'Word {i + 1}' for i in range(15)])
    
    for i in range(num_topics):
        topic_words.loc[f'Topic {i + 1}'] = words[H[i].argsort()[::-1][:15]]
    
    return nmf_model, topic_words

def assign_topics(nmf_model, dtm):
    """Assign topics to documents."""
    topic_results = nmf_model.transform(dtm)
    return topic_results.argmax(axis=1)

def save_results_to_s3(s3_client, df, bucket_name, object_key, models=None):
    """Save the results and models to S3."""
    p, _ = os.path.splitext(object_key)
    new_object_key = f"preprocessed_labeled_py/{p.split('/')[1]}.csv"
    s3_client.save_df_to_s3(df, bucket_name, new_object_key)
    
    if models:
        for model, name in models:
            s3_client.save_model_to_s3(model, bucket_name, f"{name}_model.pkl")

if __name__ == "__main__":
    s3_client = s3Loader()
    clean_bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")
    object_key = 'preprocessed_py/complaints-2021-05-14_08_16_.csv'

    data = s3_client.load_csv_from_s3(object_key=object_key, bucket_name=clean_bucket_name)
    df = pd.read_csv(io.BytesIO(data))
    df['complaint_final'] = df['complaint_final'].replace(np.nan, "")

    tfidf, dtm = vectorize_text(df['complaint_final'])
    num_topics = 5
    nmf_model, topic_words = extract_topics(dtm, num_topics, tfidf)
    df['Topic'] = assign_topics(nmf_model, dtm)

    models_to_save = [(tfidf, "tfidf"), (nmf_model, "nmf")]
    save_results_to_s3(s3_client, df, clean_bucket_name, object_key, models_to_save)