from src.config import Config
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from dotenv import load_dotenv
import os

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
    return result_df, model

def train_and_evaluate_model(train_df, test_df, feature_col, label_col):
    """Train and evaluate the Logistic Regression model."""
    lr = LogisticRegression(featuresCol=feature_col, labelCol=label_col, maxIter=10)
    model = lr.fit(train_df)
    
    # Evaluate on training data
    train_predictions = model.transform(train_df)
    train_accuracy = train_predictions.filter(train_predictions[label_col] == train_predictions.prediction).count() / float(train_df.count())
    print(f"Training Accuracy: {train_accuracy}")
    
    # Evaluate on testing data
    test_predictions = model.transform(test_df)
    test_accuracy = test_predictions.filter(test_predictions[label_col] == test_predictions.prediction).count() / float(test_df.count())
    print(f"Test Accuracy: {test_accuracy}")
    
    return model

def save_model_to_s3(model, bucket_name, model_name):
    """Save the model to an S3 bucket."""
    model.write().overwrite().save(f"s3a://{bucket_name}/models_local/{model_name}_model")

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
    object_key = 'preprocessed_labeled/complaints-2021-05-14_08_16_.parquet'
    
    # Load preprocessed data from S3
    df = spark.read.parquet(f"s3a://{clean_bucket_name}/{object_key}")
    
    # Replace NaNs with empty strings
    df = df.na.fill({'complaint_what_happened': ""})
    
    df = df.select(
        col('complaint_what_happened'),
        col('Topic'))
    
    # Apply vectorization
    df, tfidf_model = vectorize_text(df, 'complaint_what_happened', 'features')
    
    # Split data into training and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Train and evaluate model
    lr_model = train_and_evaluate_model(train_df, test_df, 'features', 'Topic')
    
    # Save models
    models_to_save = [(tfidf_model, "tfidf"), (lr_model, "logistic_regression")]
    for model, name in models_to_save:
        save_model_to_s3(model, clean_bucket_name, name)

    spark.stop()