import os
from dotenv import load_dotenv
# from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType


load_dotenv()

class Config:
    """Config for application."""
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    s3_clean_bucket: str = os.environ.get("S3_CLEAN_BUCKET_NAME")
    s3_raw_bucket: str = os.environ.get("S3_RAW_BUCKET_NAME")
    aws_access_key_id = os.environ.get("S3_ID")
    aws_secret_access_key = os.environ.get("S3_SECRET")
    aws_endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    master: str = "local[*]"
    app_name: str = "ModelInference"

    mlflow_server_host = "89.169.131.68"
    # run_id: str = "73decc58d8b642b4a7cc70bb3dfaf6f9"

    topic_mapping: dict = { 0:"Bank account services",
                            1:"Credit card / Prepaid card",
                            2:"Others",
                            3:"Theft/Dispute reporting",
                            4:"Mortgages/loans" }