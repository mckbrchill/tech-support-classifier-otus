from config import Config
from pyspark import SparkConf
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql import Row
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class InferenceModel:
    def __init__(self) -> None:
        self.config = Config()
        self.spark = self.create_spark_session()
        self.tfidf, self.lr = self.load_models()

    def create_spark_session(self):
        """Create and return a Spark session."""
        config = Config()
        conf = (
            SparkConf()
            .setMaster(config.master)
            .setAppName("TextClassificationInference")
            .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .set("spark.hadoop.fs.s3a.access.key", config.aws_access_key_id)
            .set("spark.hadoop.fs.s3a.secret.key", config.aws_secret_access_key)
            .set("spark.hadoop.fs.s3a.endpoint", config.aws_endpoint_url)
        )
        return SparkSession.builder.config(conf=conf).getOrCreate()

    def load_models(self):
        """Load and return the trained model from S3."""
        lr_path = f"s3a://{self.config.s3_clean_bucket}/models_local/logistic_regression_model"
        tfidf_path = f"s3a://{self.config.s3_clean_bucket}/models_local/tfidf_model"
        tfidf = PipelineModel.load(tfidf_path)
        lr = LogisticRegressionModel.load(lr_path)

        return tfidf, lr

    def predict(self, data):
        """Make predictions on new data passed as a dictionary or list of dictionaries."""
        if isinstance(data, dict):
            data = [data]
        inference_df = self.spark.createDataFrame([Row(**d) for d in data])
        inference_df = inference_df.fillna({'complaint_what_happened': ""})
        vectorized_df = self.tfidf.transform(inference_df)
        predictions = self.lr.transform(vectorized_df)
        result_df = predictions.select('complaint_what_happened', 'prediction')
        return result_df.collect()