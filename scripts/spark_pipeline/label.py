from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import IDFModel, CountVectorizerModel, Tokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA, LocalLDAModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import os
from dotenv import load_dotenv
import numpy as np
from config import Config

load_dotenv()

def vectorize_text(df, count_vectorizer_model, tfidf_model):
    """Vectorize text using a pre-trained CountVectorizer and TF-IDF models."""
    tokenizer = Tokenizer(inputCol='complaint_final', outputCol='words')
    remover = StopWordsRemover(inputCol='words', outputCol='filtered_words')
    
    df = tokenizer.transform(df)
    df = remover.transform(df)
    
    # Transform the text data into TF vectors
    df = count_vectorizer_model.transform(df)
    
    # Transform the TF vectors into TF-IDF vectors
    df = tfidf_model.transform(df)
    
    return df

def argmax(array):
    return int(np.argmax(array))

argmax_udf = udf(argmax, IntegerType())

def assign_topics(df, lda_model, output_col):
    """Assign topics to documents using a pre-trained LDA model."""
    transformed = lda_model.transform(df)
    return transformed.withColumn(output_col, argmax_udf(col('topicDistribution')))

def save_results_to_s3(df, bucket_name, object_key):
    """Save the results to S3."""
    p, _ = os.path.splitext(object_key)
    new_object_key = f"preprocessed_labeled/{p.split('/')[-1]}"
    df.coalesce(1).write.parquet(f"s3a://{bucket_name}/{new_object_key}", mode="overwrite")

if __name__ == "__main__":
    config = Config()
    conf = (
        SparkConf().setMaster(config.master).setAppName("Label Topics")
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
	    .set("spark.hadoop.fs.s3a.access.key", config.aws_access_key_id) \
	    .set("spark.hadoop.fs.s3a.secret.key", config.aws_secret_access_key) \
	    .set("spark.hadoop.fs.s3a.endpoint", config.aws_endpoint_url)
    )
    spark = SparkSession.builder.config(conf).getOrCreate()
    
    clean_bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")
    object_key = 'preprocessed/complaints-2021-05-14_08_16_.parquet'
    
    # Load preprocessed data from S3
    df = spark.read.parquet(f"s3a://{clean_bucket_name}/{object_key}")
    
    # Replace NaNs with empty strings
    df = df.na.fill({'complaint_final': ""})
    
    # Load the CountVectorizer, TF-IDF and LDA models from S3
    count_vectorizer_model = CountVectorizerModel.load(f"s3a://{clean_bucket_name}/models/tf_model")
    tfidf_model = IDFModel.load(f"s3a://{clean_bucket_name}/models/idf_model")
    lda_model = LocalLDAModel.load(f"s3a://{clean_bucket_name}/models/lda_model")
    
    # Vectorize text using the loaded TF-IDF model
    df = vectorize_text(df, count_vectorizer_model, tfidf_model)
    
    # Assign topics using the loaded LDA model
    df = assign_topics(df, lda_model, 'Topic')
    
    # Save the results
    save_results_to_s3(df, clean_bucket_name, object_key)
    
    spark.stop()