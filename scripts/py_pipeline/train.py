import os
import io
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from s3client import s3Loader


load_dotenv()

def load_data(s3_client, bucket_name, object_key):
    data = s3_client.load_csv_from_s3(object_key=object_key, bucket_name=bucket_name)
    return pd.read_csv(io.BytesIO(data))

def vectorize_data(data):
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(data['complaint_what_happened'])
    s3_client.save_model_to_s3(count_vect.vocabulary_, bucket_name, "count_vector.pkl")

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    s3_client.save_model_to_s3(tfidf_transformer, bucket_name, "tfidf_train.pkl")
    return X_tfidf

def train_model(X_tfidf, y):
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=42)
    logreg = LogisticRegression(random_state=42, solver='liblinear').fit(X_train, y_train)
    print(f"Model Accuracy: {logreg.score(X_test, y_test)}")

if __name__ == "__main__":
    s3_client = s3Loader()
    bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")
    object_key = 'preprocessed_labeled/complaints-2021-05-14_08_16_.csv'

    data = s3_client.load_csv_from_s3(object_key=object_key, bucket_name=bucket_name)
    df = pd.read_csv(io.BytesIO(data))

    training_data = df[['complaint_what_happened', 'Topic']]
    X_tfidf = vectorize_data(training_data)
    train_model(X_tfidf, training_data['Topic'])