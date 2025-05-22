import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import papermill as pm
from mlflow.tracking import MlflowClient

class CustomPapermillOperator(BaseOperator):
    @apply_defaults
    def __init__(self, input_path, output_path, parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.output_path = output_path
        self.parameters = parameters or {}

    def execute(self, context):
        self.log.info(f"Executing Notebook: {self.input_path}")
        pm.execute_notebook(
            self.input_path,
            self.output_path,
            parameters=self.parameters
        )
        metadata_path = "/opt/airflow/out/notebook_output_metadata.json"
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
                for key, val in metadata.items():
                    context['ti'].xcom_push(key=key, value=val)
                    self.log.info(f"Pushed XCom {key}: {val}")
        else:
            self.log.warning("No metadata file found.")

def promote_model_to_production(**context):
    client = MlflowClient("http://localhost:5001")
    model_version = context['ti'].xcom_pull(task_ids='local_mlflow_tracking', key='model_version')
    if not model_version:
        raise ValueError("No model_version found in XCom from local_mlflow_tracking.")
    client.transition_model_version_stage(
        name="HeartAttackRiskModel",
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Model version {model_version} promoted to Production.")

default_args = {
    "owner": "airflow",
    "email": ["posinarevanth@gmail.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="heart_attack_full_pipeline",
    description="Manual trigger Heart Attack Risk prediction pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    default_args=default_args,
    tags=["verikai", "heart-risk"],
) as dag:

    eda_process_data = CustomPapermillOperator(
        task_id="eda_process_data",
        input_path="/opt/project/notebooks/01_eda.ipynb",
        output_path="/opt/airflow/out/eda_{{ ts_nodash }}.ipynb",
    )

    cloud_train_sagemaker = CustomPapermillOperator(
    task_id="cloud_train_sagemaker",
    input_path="/opt/project/notebooks/02A_sagemaker_train_model.ipynb",
    output_path="/opt/airflow/out/sagemaker_train_{{ ts_nodash }}.ipynb",
    parameters={
        "bucket":   "verikai-heart-risk-pipeline",
        "prefix":   "data/processed_data",
        "role_arn": "arn:aws:iam::904233112003:role/SageMakerExecutionRole-rev",
        "region":   "us-east-1",
    }
)

    cloud_deploy_sagemaker = CustomPapermillOperator(
        task_id="cloud_deploy_sagemaker",
        input_path="/opt/project/notebooks/02B_sagemaker_deploy_model.ipynb",
        output_path="/opt/airflow/out/sagemaker_deploy_{{ ts_nodash }}.ipynb",
    )

    cloud_latency_alarm = CustomPapermillOperator(
        task_id="cloud_latency_alarm",
        input_path="/opt/project/notebooks/04_create_latency_alarm.ipynb",
        output_path="/opt/airflow/out/latency_alarm_{{ ts_nodash }}.ipynb",
        parameters={
            "endpoint_name": "{{ ti.xcom_pull(task_ids='cloud_deploy_sagemaker', key='endpoint_name') }}"
        }
    )

    cloud_hpo_analysis = CustomPapermillOperator(
        task_id="cloud_hpo_analysis",
        input_path="/opt/project/notebooks/05_hpo_analysis.ipynb",
        output_path="/opt/airflow/out/hpo_analysis_{{ ts_nodash }}.ipynb",
    )

    local_train_mlflow = CustomPapermillOperator(
        task_id="local_train_mlflow",
        input_path="/opt/project/notebooks/02_train_model.ipynb",
        output_path="/opt/airflow/out/local_train_{{ ts_nodash }}.ipynb",
    )

    local_mlflow_tracking = CustomPapermillOperator(
        task_id="local_mlflow_tracking",
        input_path="/opt/project/notebooks/03_mlflow_tracking.ipynb",
        output_path="/opt/airflow/out/mlflow_tracking_{{ ts_nodash }}.ipynb",
    )

    local_promote_model = PythonOperator(
        task_id="local_promote_model",
        python_callable=promote_model_to_production,
        provide_context=True,
    )

    # DAG Task Dependencies
    eda_process_data >> cloud_train_sagemaker >> cloud_deploy_sagemaker >> cloud_latency_alarm >> cloud_hpo_analysis
    eda_process_data >> local_train_mlflow >> local_mlflow_tracking >> local_promote_model
