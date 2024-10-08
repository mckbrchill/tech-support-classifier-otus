from src.config import Config
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from dotenv import load_dotenv
import os
import mlflow
import mlflow.spark
import numpy as np
from scipy.stats import norm
from datetime import datetime
from mlflow.tracking import MlflowClient

load_dotenv()

def vectorize_text(df, input_col, output_col):
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    tf = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
    idf = IDF(inputCol="raw_features", outputCol=output_col)
    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf])
    model = pipeline.fit(df)
    result_df = model.transform(df)
    return result_df, model

def train_and_evaluate_model(train_df, test_df, feature_col, label_col):
    lr = LogisticRegression(featuresCol=feature_col, labelCol=label_col, maxIter=10)
    model = lr.fit(train_df)
    
    train_predictions = model.transform(train_df)
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
    train_accuracy = evaluator.evaluate(train_predictions, {evaluator.metricName: "accuracy"})
    test_predictions = model.transform(test_df)
    test_accuracy = evaluator.evaluate(test_predictions, {evaluator.metricName: "accuracy"})
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    
    return model

def bootstrap_metrics(test_df, model, label_col='Topic', num_samples=5):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for _ in range(num_samples):
        sample_df = test_df.sample(withReplacement=True, fraction=0.1, seed=np.random.randint(0, 10000))
        predictions = model.transform(sample_df)
        metrics['accuracy'].append(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}))
        metrics['precision'].append(evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}))
        metrics['recall'].append(evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"}))
        metrics['f1'].append(evaluator.evaluate(predictions, {evaluator.metricName: "f1"}))
    return metrics

def calculate_z_test(new_metrics, prev_metrics):
    z_scores = {}
    p_values = {}
    for metric in new_metrics:
        new_mean = np.mean(new_metrics[metric])
        new_std = np.std(new_metrics[metric])
        prev_mean = prev_metrics[metric][0]
        prev_std = prev_metrics[metric][1]
        z_score = (new_mean - prev_mean) / np.sqrt(new_std**2 + prev_std**2)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        z_scores[metric] = z_score
        p_values[metric] = p_value
    return z_scores, p_values

def log_ab_test_results(new_metrics, prev_metrics, z_scores, p_values, run_id, alpha=0.05):
    for metric in new_metrics:
        mlflow.log_metric(f"new_{metric}_mean", np.mean(new_metrics[metric]))
        mlflow.log_metric(f"new_{metric}_std", np.std(new_metrics[metric]))
        mlflow.log_metric(f"new_{metric}_2.5th_percentile", np.percentile(new_metrics[metric], 2.5))
        mlflow.log_metric(f"new_{metric}_97.5th_percentile", np.percentile(new_metrics[metric], 97.5))
        if prev_metrics:
            p_value = p_values[metric]
            z_score = z_scores[metric]
            mlflow.log_metric(f"{metric}_z_score", z_score)
            mlflow.log_metric(f"{metric}_p_value", p_value)
            if p_value < alpha:
                mlflow.log_text(
                    f"{metric} has statistically significant difference (p_value={p_value}, alpha={alpha})",
                    f"a_b_test_results_for_{metric}.txt")
            else:
                mlflow.log_text(
                    f"{metric} does not have statistically significant difference (p_value={p_value}, alpha={alpha})",
                    f"a_b_test_results_for_{metric}.txt")

def ab_test(test_df, model, run_id, client, experiment_id):
    new_metrics = bootstrap_metrics(test_df, model)
    prev_runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=2)
    if len(prev_runs) > 1:
        prev_run_id = prev_runs[1].info.run_id
        prev_metrics = {}
        for metric in new_metrics:
            prev_metrics[metric] = (client.get_metric_history(prev_run_id, f"new_{metric}_mean")[0].value,
                                    client.get_metric_history(prev_run_id, f"new_{metric}_std")[0].value)
        z_scores, p_values = calculate_z_test(new_metrics, prev_metrics)
        log_ab_test_results(new_metrics, prev_metrics, z_scores, p_values, run_id)
    else:
        prev_metrics = {}
        z_scores = {}
        p_values = {}
        log_ab_test_results(new_metrics, prev_metrics, z_scores, p_values, run_id)

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
    
    mlflow.set_tracking_uri(f"http://{config.mlflow_server_host}:8000")
    run_name = 'LogisticRegressionModelRun' + ' ' + str(datetime.now())
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ab_pyspark_experiment")
    experiment_id = experiment.experiment_id
    
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        df = spark.read.parquet(f"s3a://{clean_bucket_name}/{object_key}")
        df = df.na.fill({'complaint_what_happened': ""})
        df = df.select(col('complaint_what_happened'), col('Topic'))
        
        df, tfidf_model = vectorize_text(df, 'complaint_what_happened', 'features')
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
        lr_model = train_and_evaluate_model(train_df, test_df, 'features', 'Topic')
        run_id = mlflow.active_run().info.run_id
        ab_test(test_df, lr_model, run_id, client, experiment_id)
        
        # Save models to MLflow and S3
        mlflow.spark.save_model(lr_model, "models_local/lr_model")
        mlflow.spark.save_model(tfidf_model, "models_local/tfidf_model")

        # Save models
        models_to_save = [(tfidf_model, "tfidf"), (lr_model, "logistic_regression")]
        for model, name in models_to_save:
            save_model_to_s3(model, clean_bucket_name, name)

    
    spark.stop()