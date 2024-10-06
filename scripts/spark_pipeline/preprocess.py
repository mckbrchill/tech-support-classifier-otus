import os
import re
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when,  regexp_replace
from pyspark.sql.types import StringType
import spacy
from textblob import TextBlob
from dotenv import load_dotenv
from config import Config
import nltk
import sys

# import findspark
# findspark.init()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Load necessary NLTK data
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
os.system('python -m textblob.download_corpora')

# Load environment variables
load_dotenv()

# Define UDFs
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

def preprocess_complaints(df):
    """Preprocess the complaints data using Spark UDFs."""

    # Select and alias nested fields within _score
    df = df.select(
        col('_id').alias('id'),
        col('_index').alias('index'),
        col('_score').alias('score'),
        col('_type').alias('type'),
        col('_source.tags').alias('tags'),
        col('_source.zip_code').alias('zip_code'),
        col('_source.complaint_id').alias('complaint_id'),
        col('_source.issue').alias('issue'),
        col('_source.date_received').alias('date_received'),
        col('_source.state').alias('state'),
        col('_source.consumer_disputed').alias('consumer_disputed'),
        col('_source.product').alias('product'),
        col('_source.company_response').alias('company_response'),
        col('_source.company').alias('company'),
        col('_source.submitted_via').alias('submitted_via'),
        col('_source.date_sent_to_company').alias('date_sent_to_company'),
        col('_source.company_public_response').alias('company_public_response'),
        col('_source.sub_product').alias('sub_product'),
        col('_source.timely').alias('timely'),
        col('_source.complaint_what_happened').alias('complaint_what_happened'),
        col('_source.sub_issue').alias('sub_issue'),
        col('_source.consumer_consent_provided').alias('consumer_consent_provided')
    )
    
    df = df.withColumn('complaint_what_happened', col('complaint_what_happened').cast(StringType()))
    df = df.withColumn(
        'complaint_what_happened',
        when(col('complaint_what_happened') == '', None).otherwise(col('complaint_what_happened'))
    )
    df = df.na.drop(subset=['complaint_what_happened'])

    clean_udf = udf(clean_text, StringType())
    lemmatize_udf = udf(lemmatize_text, StringType())
    pos_tags_udf = udf(get_pos_tags, StringType())

    df = df.na.drop(subset=['complaint_what_happened'])
    
    df = df.withColumn('complaint_cleaned', clean_udf(col('complaint_what_happened')))
    df = df.withColumn('complaint_lemmatized', lemmatize_udf(col('complaint_cleaned')))
    df = df.withColumn('complaint_pos_filtered', pos_tags_udf(col('complaint_lemmatized')))
    df = df.withColumn('complaint_final', 
                   regexp_replace(
                       regexp_replace(col('complaint_pos_filtered'), '-PRON-', ''), 
                       'xxxx', ''))

    return df

if __name__ == "__main__":
    object_key = 'complaints-2021-05-14_08_16_.json'
    
    config = Config()

    conf = (
        SparkConf().setMaster(config.master).setAppName(config.app_name)
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
	    .set("spark.hadoop.fs.s3a.access.key", config.aws_access_key_id) \
	    .set("spark.hadoop.fs.s3a.secret.key", config.aws_secret_access_key) \
	    .set("spark.hadoop.fs.s3a.endpoint", config.aws_endpoint_url) \
        .set('spark.default.parallelism', 1) \
        .set('spark.sql.shuffle.partitions', 1) \
        .set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
    )

    # Initialize Spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    nlp = spacy.load("en_core_web_sm")

    # S3 bucket details
    raw_bucket_name = os.environ.get("S3_RAW_BUCKET_NAME")
    clean_bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")

    # Load data from S3
    s3_filepath = f"s3a://{raw_bucket_name}/{object_key}"
    df = spark.read.json(s3_filepath)

    # Preprocess data
    preprocessed_df = preprocess_complaints(df)

    # Define output path
    p, _ = os.path.splitext(object_key)
    new_object_key = f"preprocessed/{p}.parquet"
    output_path = f"s3a://{clean_bucket_name}/{new_object_key}"

    # Write preprocessed data back to S3
    preprocessed_df.coalesce(1).write.parquet(output_path, mode="overwrite")

    # Stop Spark session
    spark.stop()