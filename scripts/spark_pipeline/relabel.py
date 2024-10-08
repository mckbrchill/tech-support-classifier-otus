from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import IDF, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import os
from dotenv import load_dotenv
import numpy as np
from src.config import Config

load_dotenv()

def vectorize_text(df, input_col, output_col):
    """Vectorize text using TF-IDF."""
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    tf = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
    idf = IDF(inputCol="raw_features", outputCol=output_col)
    
    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf])
    model = pipeline.fit(df)
    result_df = model.transform(df)
    return result_df, model.stages[-2], model.stages[-1]

def extract_topics(df, input_col, num_topics):
    """Extract topics using Spark LDA."""
    lda = LDA(k=num_topics, seed=40, featuresCol=input_col)
    lda_model = lda.fit(df)
    topics = lda_model.describeTopics(15)
    return lda_model, topics


def argmax(array):
    return int(np.argmax(array))

argmax_udf = udf(argmax, IntegerType())

def assign_topics(df, lda_model, input_col, output_col):
    """Assign topics to documents."""
    transformed = lda_model.transform(df)
    return transformed.withColumn(output_col, argmax_udf(col('topicDistribution')))

def save_results_to_s3(df, bucket_name, object_key, models=None):
    """Save the results and models to S3."""
    p, _ = os.path.splitext(object_key)
    new_object_key = f"preprocessed_labeled/{p.split('/')[-1]}"
    df.coalesce(1).write.parquet(f"s3a://{bucket_name}/{new_object_key}", mode="overwrite")
    if models:
        for model, name in models:
            model.write().overwrite().save(f"s3a://{bucket_name}/models/{name}_model")

if __name__ == "__main__":
    config = Config()
    conf = (
        SparkConf().setMaster(config.master).setAppName("Relabel Topics")
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
	    .set("spark.hadoop.fs.s3a.access.key", config.aws_access_key_id) \
	    .set("spark.hadoop.fs.s3a.secret.key", config.aws_secret_access_key) \
	    .set("spark.hadoop.fs.s3a.endpoint", config.aws_endpoint_url)
    )
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    clean_bucket_name = os.environ.get("S3_CLEAN_BUCKET_NAME")
    object_key = 'preprocessed/complaints-2021-05-14_08_16_.parquet'

    df = spark.read.parquet(f"s3a://{clean_bucket_name}/{object_key}")
    df = df.na.fill({'complaint_final': ""})
    df, tf_model, idf_model = vectorize_text(df, 'complaint_final', 'features')

    num_topics = 5
    lda_model, topic_words = extract_topics(df, 'features', num_topics)

    df = assign_topics(df, lda_model, 'features', 'Topic')
    
    models_to_save = [(tf_model, "tf"), (idf_model, "idf"), (lda_model, "lda")]
    save_results_to_s3(df, clean_bucket_name, object_key, models_to_save)
    
    spark.stop()