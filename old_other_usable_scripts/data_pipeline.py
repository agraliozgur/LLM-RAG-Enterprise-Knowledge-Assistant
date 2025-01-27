"""
Airflow DAG: Data ingestion -> Preprocessing -> Embedding -> Qdrant Upsert
Assumes you already have a Qdrant server running at localhost:6333
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import json

# Sample modules we'll define below
from my_project.data_preprocessing import collect_documents, preprocess_documents
from my_project.embedding_upload import embed_and_upsert

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='data_ingestion_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',  # run once a week, for example
    catchup=False
) as dag:

    t1_collect = PythonOperator(
        task_id='collect_documents',
        python_callable=collect_documents,
        op_kwargs={
            "source_folder": "/data/incoming_docs",  # adjust paths
            "output_folder": "/data/raw_docs"
        }
    )

    t2_preprocess = PythonOperator(
        task_id='preprocess_documents',
        python_callable=preprocess_documents,
        op_kwargs={
            "input_folder": "/data/raw_docs",
            "output_file": "/data/preprocessed.jsonl"
        }
    )

    t3_embed_upsert = PythonOperator(
        task_id='embed_and_upsert_to_qdrant',
        python_callable=embed_and_upsert,
        op_kwargs={
            "jsonl_path": "/data/preprocessed.jsonl",
            "collection_name": "enterprise_chunks",
            "qdrant_url": "http://qdrant:6333"
        }
    )

    t1_collect >> t2_preprocess >> t3_embed_upsert
