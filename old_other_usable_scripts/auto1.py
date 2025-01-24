from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

def collect_data(**kwargs):
    pass

def generate_embeddings(**kwargs):
    pass

def test_inference(**kwargs):
    pass

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('rag_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

    t1 = PythonOperator(
        task_id='collect_data',
        python_callable=collect_data
    )
    t2 = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings
    )
    t3 = PythonOperator(
        task_id='test_inference',
        python_callable=test_inference
    )

    t1 >> t2 >> t3
