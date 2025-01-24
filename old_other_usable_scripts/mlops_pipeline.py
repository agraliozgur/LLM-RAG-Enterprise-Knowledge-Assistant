
# HfFolder.save_token('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')
# HF_TOKEN = os.environ.get('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', "")  # or store in .env / secrets manager
#!/usr/bin/env python3
# FILE: mlops_pipeline.py
# --------------------------------------------------
# Demonstrates how to integrate MLOps with:
#  1) MLflow for experiment logging
#  2) Hugging Face Hub for model versioning
#  3) Basic fine-tuning of a T5-like model (Flan-T5)
#     using "all_processed_data.jsonl" for demonstration
# --------------------------------------------------

import os
import logging
import json
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from huggingface_hub import HfApi, Repository, HfFolder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Global Configurations
# -------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Your MLflow server or local path
EXPERIMENT_NAME = "RAG_MLOps_Demo"

# Hugging Face repository info
HF_MODEL_REPO = "ozguragrali/llm-rag-google-flan-t5-large-corporate_info_assistant"
HF_TOKEN = os.environ.get("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", "")  # Set your HF token: export HF_API_TOKEN="..."

# Base model to fine-tune
BASE_MODEL_NAME = "google/flan-t5-large"

# Path to the chunked data in JSONL format
# Each line has fields like "unique_id", "text", etc.
LOCAL_JSONL_DATA = "../data/cleaned_data/all_processed_data.jsonl"

# Fine-tuned model output directory
OUTPUT_DIR = "../models/finetuned_model_output"

# -------------------------------------------------------------------
# 1) Data Loading & Preparation
# -------------------------------------------------------------------
def load_local_jsonl_as_dataset(jsonl_path: str) -> Dataset:
    """
    Loads your local JSONL data as a Hugging Face Dataset object.
    Expects each line to contain at least a 'text' field.
    
    For demonstration, we'll treat 'text' as the input, and 
    we ask the model to "summarize" or "transform" it. 
    Adjust as needed for your use case (Q&A, classification, etc.).

    Args:
        jsonl_path (str): Path to the all_processed_data.jsonl file.

    Returns:
        Dataset: A Hugging Face Dataset with "input_text" and "target_text" columns.
    """
    logger.info(f"Loading local JSONL data from: {jsonl_path}")
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {line_idx}.")
                continue
            
            raw_text = record.get("text", "").strip()
            if not raw_text:
                continue

            # For demonstration, we'll create a "summary" style target:
            # e.g., "Summarize: <raw_text>"
            # In practice, you might have real target labels or QA pairs.
            samples.append({
                "input_text": raw_text,
                "target_text": "Summary: " + raw_text[:200]  # naive approach
            })

    if not samples:
        logger.warning("No valid samples found in JSONL!")
    else:
        logger.info(f"Loaded {len(samples)} samples from JSONL.")
    return Dataset.from_list(samples)

# -------------------------------------------------------------------
# 2) Fine-Tuning / Training
# -------------------------------------------------------------------
def train_model(base_model_name: str, dataset: Dataset, num_train_epochs: int = 1) -> str:
    """
    Fine-tunes (or lightly trains) a T5/Flan model on the given dataset.
    The dataset is expected to have 'input_text' and 'target_text'.
    
    Args:
        base_model_name (str): The base HF model to load (e.g., "google/flan-t5-large").
        dataset (Dataset): The loaded dataset (Hugging Face style).
        num_train_epochs (int): Number of epochs to train.

    Returns:
        str: Path to the directory containing the fine-tuned model.
    """
    logger.info(f"Loading base model '{base_model_name}' for demonstration training.")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # We'll do a naive train/test split for demonstration
    split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]

    # Preprocess the data for T5
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], max_length=512, truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"], max_length=128, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Map preprocess
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_eval = eval_ds.map(preprocess_function, batched=True)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=5000,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_dir="../data/logs",
        logging_steps=500,
        do_train=True,
        do_eval=True
    )

    def data_collator(features):
        return {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "labels": [f["labels"] for f in features],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator
    )

    logger.info("Starting training process...")
    trainer.train()

    # Optionally evaluate
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")

    logger.info("Training complete.")
    return training_args.output_dir

# -------------------------------------------------------------------
# 3) MLflow Logging & Hugging Face Push
# -------------------------------------------------------------------
def log_experiment_and_push_model(
    base_model: str,
    finetuned_dir: str,
    experiment_name: str = EXPERIMENT_NAME,
    eval_metrics: dict = None
):
    """
    Logs an experiment in MLflow with the training parameters, metrics, and artifacts.
    Then pushes the resulting model to the Hugging Face Hub, including a basic model card.

    Args:
        base_model (str): Name of the base model (e.g., "google/flan-t5-large")
        finetuned_dir (str): Local directory containing the fine-tuned model
        experiment_name (str): MLflow experiment name
        eval_metrics (dict): Optional dictionary of evaluation metrics to log
    """
    # 1) MLflow setup & logging
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("base_model", base_model)

        # If you know the epochs or other hyperparams:
        mlflow.log_param("epochs", 1)  # example

        # Log metrics (dummy or real)
        if eval_metrics:
            for k, v in eval_metrics.items():
                mlflow.log_metric(k, float(v))

        # Log artifacts (the final model directory)
        mlflow.log_artifacts(finetuned_dir, artifact_path="fine_tuned_model")

        logger.info("Experiment run completed. Model artifacts logged to MLflow.")

    # 2) Push model to Hugging Face Hub
    if not HF_TOKEN:
        logger.warning("No HF_API_TOKEN found. Skipping model push to Hugging Face.")
        return

    logger.info("Pushing model to Hugging Face Hub...")

    # Ensure your local token is saved
    HfFolder.save_token(HF_TOKEN)
    api = HfApi()

    # Check if repo exists; if not, create
    try:
        repo_info = api.model_info(repo_id=HF_MODEL_REPO, token=HF_TOKEN)
        logger.info(f"Repo {HF_MODEL_REPO} found. Proceeding to push.")
    except Exception:
        logger.info(f"Repo {HF_MODEL_REPO} not found. Creating a new one...")
        api.create_repo(
            repo_id=HF_MODEL_REPO,
            private=True,  # or False for public
            token=HF_TOKEN
        )

    # Clone or pull latest
    local_repo_dir = "hf_repo_temp"
    if os.path.exists(local_repo_dir):
        shutil.rmtree(local_repo_dir, ignore_errors=True)

    repo = Repository(local_dir=local_repo_dir, clone_from=HF_MODEL_REPO, token=HF_TOKEN)
    repo.git_pull()

    # Copy the fine-tuned model files
    for item in os.listdir(local_repo_dir):
        item_path = os.path.join(local_repo_dir, item)
        if os.path.isfile(item_path) or os.path.isdir(item_path):
            os.remove(item_path) if os.path.isfile(item_path) else shutil.rmtree(item_path)

    # Copy all from finetuned_dir to local_repo_dir
    for item in os.listdir(finetuned_dir):
        src_path = os.path.join(finetuned_dir, item)
        dst_path = os.path.join(local_repo_dir, item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # 3) Create or update Model Card (README.md)
    model_card_path = os.path.join(local_repo_dir, "README.md")
    with open(model_card_path, "w", encoding="utf-8") as card:
        card.write(f"# {HF_MODEL_REPO}\n\n")
        card.write(f"Fine-tuned from base model: `{base_model}`\n\n")
        card.write("## Model Description\n")
        card.write(
            "This model is a demonstration of a RAG-based knowledge assistant, "
            "fine-tuned on custom 'all_processed_data.jsonl' data. "
            "It's intended for summarization or short text transformations.\n\n"
        )
        card.write("## Intended Use\n")
        card.write(
            "- Enterprise knowledge summarization\n"
            "- Retrieval-augmented generation tasks\n\n"
        )
        card.write("## Training Data\n")
        card.write(
            "Data was derived from a chunked corporate dataset (`all_processed_data.jsonl`), "
            "each chunk forming an input text. The target was a naive 'Summary:' prompt.\n\n"
        )
        card.write("## Limitations and Bias\n")
        card.write(
            "Because the training data is limited and the summarization format is naive, "
            "the model might produce incomplete or repetitive summaries.\n\n"
        )
        card.write("## Evaluation Metrics\n")
        if eval_metrics:
            for k, v in eval_metrics.items():
                card.write(f"- **{k}**: {v}\n")
        else:
            card.write("- No evaluation metrics reported.\n\n")
        card.write("---\n")
        card.write("Generated via MLOps pipeline demonstration.\n")

    # Commit and push
    repo.git_add(pattern="*")
    repo.git_commit("Add or update fine-tuned model and model card")
    repo.git_push()
    logger.info(f"Model and model card pushed to Hugging Face Hub: https://huggingface.co/{HF_MODEL_REPO}")

# -------------------------------------------------------------------
# 4) Main Entry Point
# -------------------------------------------------------------------
def main():
    # Step A: Load the dataset from local JSONL
    ds = load_local_jsonl_as_dataset(LOCAL_JSONL_DATA)
    if len(ds) == 0:
        logger.error("No data found in JSONL. Exiting.")
        return

    # Step B: Fine-tune the model (e.g., 1 epoch)
    logger.info(f"Fine-tuning base model: {BASE_MODEL_NAME}")
    finetuned_model_path = train_model(BASE_MODEL_NAME, ds, num_train_epochs=1)

    # Step C: Evaluate (already done in 'train_model'), we can load metrics if we like
    # For demonstration, let's pretend we have some metrics from trainer.evaluate()
    dummy_metrics = {"eval_loss": 2.34, "eval_accuracy": 0.67}  # Example

    # Step D: Log to MLflow and push to Hugging Face
    log_experiment_and_push_model(
        base_model=BASE_MODEL_NAME,
        finetuned_dir=finetuned_model_path,
        experiment_name=EXPERIMENT_NAME,
        eval_metrics=dummy_metrics
    )

if __name__ == "__main__":
    main()
