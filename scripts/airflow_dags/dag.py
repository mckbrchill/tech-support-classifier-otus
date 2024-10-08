import uuid
from datetime import datetime, timedelta

from airflow import DAG, settings
from airflow.models import Connection, Variable
from airflow.providers.yandex.operators.yandexcloud_dataproc import (
    DataprocCreateClusterOperator, DataprocCreatePysparkJobOperator,
    DataprocDeleteClusterOperator)
from airflow.utils.trigger_rule import TriggerRule

# Common settings for your environment
YC_DP_FOLDER_ID = 'b1gh2a0389gpv6i6jft2'
YC_DP_SUBNET_ID = 'e9be88k96f0kboef1313'
YC_DP_SA_ID = 'ajet88bcmvjfhvd3rhpr'
YC_DP_AZ = 'ru-central1-a'
YC_DP_SSH_PUBLIC_KEY = Variable.get("SSH_PUBLIC")
YC_DP_GROUP_ID = 'enpcamk47upe22rajimg'


# Settings for S3 buckets
YC_INPUT_DATA_BUCKET = 'customer-sup-tickets-raw/airflow/'  # YC S3 bucket for input data
YC_SOURCE_BUCKET = 'customer-sup-tickets-clean'     # YC S3 bucket for pyspark source files
YC_DP_LOGS_BUCKET = 'customer-sup-tickets-raw/airflow_logs/'      # YC S3 bucket for Data Proc cluster logs


# Создание подключения для Object Storage
session = settings.Session()
ycS3_connection = Connection(
    conn_id='yc-s3',
    conn_type='s3',
    host='https://storage.yandexcloud.net/',
    extra={
        "aws_access_key_id": Variable.get("S3_KEY_ID"),
        "aws_secret_access_key": Variable.get("S3_SECRET_KEY"),
        "host": "https://storage.yandexcloud.net/"
    }
)

if not session.query(Connection).filter(Connection.conn_id == ycS3_connection.conn_id).first():
    session.add(ycS3_connection)
    session.commit()

ycSA_connection = Connection(
    conn_id='yc-SA',
    conn_type='yandexcloud',
    extra={
        "extra__yandexcloud__public_ssh_key": Variable.get("DP_PUBLIC_SSH_KEY"),
        "extra__yandexcloud__service_account_json_path": Variable.get("DP_SA_PATH")
    }
)

if not session.query(Connection).filter(Connection.conn_id == ycSA_connection.conn_id).first():
    session.add(ycSA_connection)
    session.commit()

props = {'spark.submit.deployMode': 'cluster',
        'spark.yarn.dist.archives': f's3a://customer-sup-tickets-clean/scripts/venv.tar.gz#pyspark_venv',
        'spark.yarn.appMasterEnv.PYSPARK_PYTHON': './pyspark_venv/bin/python',
        'spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON': './pyspark_venv/bin/python',
        'spark.yarn.appMasterEnv.S3_ID': Variable.get("S3_KEY_ID"),
        'spark.yarn.appMasterEnv.S3_SECRET': Variable.get("S3_SECRET_KEY"),
        'spark.yarn.appMasterEnv.S3_RAW_BUCKET_NAME': Variable.get("S3_RAW_BUCKET_NAME"),
        'spark.yarn.appMasterEnv.S3_CLEAN_BUCKET_NAME': Variable.get("S3_CLEAN_BUCKET_NAME"),
        'spark.yarn.appMasterEnv.S3_ENDPOINT_URL': Variable.get("S3_ENDPOINT_URL")}


# Настройки DAG
with DAG(
        dag_id = 'DATA_PREPROCESS_2',
        start_date=datetime(year = 2024,month = 10,day = 8, hour=12, minute=30,second=0),
        schedule_interval = timedelta(minutes=60), # для дебага
        # schedule_interval = timedelta(days=7), 
        catchup=False
) as ingest_dag:

    create_spark_cluster = DataprocCreateClusterOperator(
        task_id='dp-cluster-create-task',
        folder_id=YC_DP_FOLDER_ID,
        cluster_name=f'tmp-dp-{uuid.uuid4()}',
        cluster_description='Temporary cluster for Spark processing under Airflow orchestration',
        subnet_id=YC_DP_SUBNET_ID,
        s3_bucket=YC_DP_LOGS_BUCKET,
        service_account_id=YC_DP_SA_ID,
        ssh_public_keys=YC_DP_SSH_PUBLIC_KEY,
        zone=YC_DP_AZ,
        cluster_image_version='2.0.43',
        masternode_resource_preset='s3-c4-m16',
        masternode_disk_type='network-ssd',
        masternode_disk_size=40,
        datanode_resource_preset='s3-c4-m16',
        datanode_disk_type='network-ssd',
        datanode_disk_size=50,
        datanode_count=1,
        services=['YARN', 'SPARK', 'HDFS', 'MAPREDUCE'],  
        computenode_count=0,           
        connection_id=ycSA_connection.conn_id,
        dag=ingest_dag,
    )

    # 2 этап: запуск задания PySpark
    spark_preprocessing = DataprocCreatePysparkJobOperator(
        task_id='dp-cluster-pyspark-preproc',
        main_python_file_uri='s3a://customer-sup-tickets-clean/scripts/preprocess.py',
        python_file_uris=['s3a://customer-sup-tickets-clean/scripts/config.py'],
        connection_id = ycSA_connection.conn_id,
        properties = props,
        dag=ingest_dag
    )

    spark_labelling = DataprocCreatePysparkJobOperator(
        task_id='dp-cluster-pyspark-label',
        main_python_file_uri='s3a://customer-sup-tickets-clean/scripts/label.py',
        python_file_uris=['s3a://customer-sup-tickets-clean/scripts/config.py'],
        connection_id = ycSA_connection.conn_id,
        properties = props,
        dag=ingest_dag
    )
    
    spark_training = DataprocCreatePysparkJobOperator(
        task_id='dp-cluster-pyspark-training',
        main_python_file_uri='s3a://customer-sup-tickets-clean/scripts/train_ab_mlflow.py',
        python_file_uris=['s3a://customer-sup-tickets-clean/scripts/config.py'],
        connection_id = ycSA_connection.conn_id,
        properties = props,
        dag=ingest_dag
    )

    delete_spark_cluster = DataprocDeleteClusterOperator(
        task_id='dp-cluster-delete-task',
        trigger_rule=TriggerRule.ALL_DONE,
        dag=ingest_dag
    )
    # Формирование DAG из указанных выше этапов
    create_spark_cluster >> spark_preprocessing >> spark_labelling >> spark_training >> delete_spark_cluster