import os
from dotenv import load_dotenv
# from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType


load_dotenv()

class Config:
    """Config for application."""
    # backend_host: str = "0.0.0.0"
    # backend_port: int = 8000

    # current_dir = Path(__file__).resolve().parent.parent
    # model_path = os.path.join(current_dir, 'model', 'sparkml')
    # schema = StructType([
    #     StructField("transaction_id", IntegerType(), True),
    #     StructField("tx_datetime", TimestampType(), True),
    #     StructField("customer_id", IntegerType(), True),
    #     StructField("terminal_id", IntegerType(), True),
    #     StructField("tx_amount", DoubleType(), True),
    #     StructField("tx_time_seconds", IntegerType(), True),
    #     StructField("tx_time_days", IntegerType(), True),
    # ])
    # vector_assembler_features: list =  [
    #     'customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds',
    #     'tx_time_days'
    # ]
    # vector_assembler_output_col: str = 'features'

    s3_clean_bucket: str = os.environ.get("S3_CLEAN_BUCKET_NAME")
    s3_raw_bucket: str = os.environ.get("S3_RAW_BUCKET_NAME")
    aws_access_key_id = os.environ.get("S3_ID")
    aws_secret_access_key = os.environ.get("S3_SECRET")
    aws_endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    master: str = "local[*]"
    app_name: str = "ModelInference"

    mlflow_server_host = "89.169.131.68"
    # run_id: str = "73decc58d8b642b4a7cc70bb3dfaf6f9"