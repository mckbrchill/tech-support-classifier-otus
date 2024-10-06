import os
import json
import joblib
import s3fs
from io import StringIO
import tempfile

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv

TIMEOUT = 180
CONFIG = BotoConfig(connect_timeout=TIMEOUT, retries={"mode": "adaptive", 'max_attempts': 5},
                     tcp_keepalive=True)
load_dotenv()

class s3Loader:
    def __init__(self,
                 source_bucket = os.environ.get("S3_RAW_BUCKET_NAME"),
                 source_prefix = "",
                 destination_bucket = os.environ.get("S3_CLEAN_BUCKET_NAME"),
                 destination_prefix = ""):
        self.source_bucket = source_bucket
        self.source_prefix = source_prefix
        self.destination_bucket = destination_bucket
        self.destination_prefix = destination_prefix

        self.s3 = boto3.client('s3',
                        aws_access_key_id=os.environ.get("S3_ID"),
                        aws_secret_access_key=os.environ.get("S3_SECRET"),
                        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
                        config=CONFIG)


    def copy_s3_objects(self):
        response = self.s3.list_objects_v2(
            Bucket=self.source_bucket,
            Prefix=self.source_prefix
        )

        for obj in response.get('Contents', []):
            source_key = obj['Key']

            if source_key.endswith('.txt'):
                destination_key = source_key.replace(self.source_prefix, self.destination_prefix, 1)

                copy_source = {
                    'Bucket':self.source_bucket,
                    'Key': source_key,
                    # 'ACL': 'public-read'
                }

                self.s3.copy_object(
                    CopySource=copy_source,
                    Bucket=self.destination_bucket,
                    Key=destination_key
                )
                print(f"Source {source_key} was successfully copied to {self.destination_bucket} with prefix {source_prefix}")

        print("Data copied successfully!")

    def upload_json_files_to_s3(self, local_directory, bucket, endpoint_url="https://storage.yandexcloud.net"):        
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                if file.endswith('.json'):
                    local_path = os.path.join(root, file)
                    s3_path = file
                    
                    try:
                        self.s3.upload_file(local_path, bucket, s3_path)
                        print(f"Uploaded {file} to {bucket}")
                    except ClientError as e:
                        print(f"Failed to upload {file}: {e}")

        print("All JSON files uploaded successfully!")

    def upload_json_files_to_s3(self, local_directory, bucket, file_extenstions = (".json"), endpoint_url="https://storage.yandexcloud.net"):
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                if file.endswith(file_extenstions):
                    local_path = os.path.join(root, file)
                    s3_path = file
                    
                    try:
                        self.s3.upload_file(local_path, bucket, s3_path)
                        print(f"Uploaded {file} to {bucket}")
                    except ClientError as e:
                        print(f"Failed to upload {file}: {e}")

        print("All JSON files uploaded successfully!")

    def load_csv_from_s3(self, object_key, bucket_name):
        response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        data = response['Body'].read()
    
        return data
    
    def load_json_from_s3(self, object_key, bucket_name):
        response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        data = json.loads(response['Body'].read().decode('utf-8'))

        return data
    
    def save_model_to_s3(self, model, bucket_name, model_name):
        """Uploads ml model to an S3 bucket."""
        # Serialize the model using joblib
        filepath = './local_model.pkl'
        joblib.dump(model, filepath)

        key = "models/" + model_name
        with open(filepath, 'rb') as model_file:
            model_data = model_file.read()
            self.s3.put_object(Bucket=bucket_name, Key=key, Body=model_data)

        os.remove(filepath)

    def load_model_from_s3(self, bucket_name, model_name):
        """Downloads ml model from an S3 bucket and loads it."""
        key = "models/" + model_name
        # Create a temporary file to store the downloaded model
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filepath = temp_file.name
            # Get the model object from S3
            response = self.s3.get_object(Bucket=bucket_name, Key=key)
            model_data = response['Body'].read()

            # Write the model data to the temporary file
            with open(temp_filepath, 'wb') as temp_model_file:
                temp_model_file.write(model_data)

        # Load the model using joblib
        model = joblib.load(temp_filepath)

        # Clean up the temporary file
        os.remove(temp_filepath)
        
        return model

    def save_df_to_s3(self, df, bucket_name, file_name):
        """Uploads DF to an S3 bucket."""
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        self.s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())


# if __name__ == "__main__":
#     s3_key_id = os.environ.get("S3_ID")
#     s3_secret = os.environ.get("S3_SECRET")
#     endpoint_url = os.environ.get("S3_ENDPOINT_URL")
#     source_bucket = os.environ.get("S3_RAW_BUCKET_NAME")
#     destination_bucket = os.environ.get("S3_CLEAN_BUCKET_NAME")

#     source_prefix = ""
#     destination_prefix = source_prefix

#     local_directory = os.environ.get("DATA_PATH")
#     upload_json_files_to_s3(local_directory, source_bucket, endpoint_url=endpoint_url)
    