import os
import random
from pathlib import Path
import numpy as np

import torch
from huggingface_hub import  HfFolder
from datasets import load_dataset, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1) Configure your Hugging Face settings
#    - Replace these with your personal/organization info and desired repo names.
HF_USERNAME = "ozguragrali"   # e.g. "john-doe" or "my-org"
DATASET_REPO_NAME = "enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large"     # Dataset repo name on HF

# 2) Paths to local data
DATA_FILE = str(_PROJECT_ROOT / "data" / "cleaned_data" / "all_processed_data.jsonl")

# 3) Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
HF_TOKEN = os.environ.get('HF_TOKEN')

def main():
    # ------------------------------------------------------------
    # A. Load & Push Dataset to Hugging Face
    # ------------------------------------------------------------
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN before publishing a dataset.")
    HfFolder.save_token(HF_TOKEN)
    print("Loading local JSONL dataset...")
    # Assuming your JSONL has fields like {"input": "some text", "target": "some text"}
    raw_dataset = load_dataset(
        "json",
        data_files=DATA_FILE,
        split="train"  # single split
    )
    print(f"Loaded {len(raw_dataset)} records from {DATA_FILE}")

    # Split into train/validation (e.g., 80%/20%)
    ds_split = raw_dataset.train_test_split(test_size=0.2, seed=SEED)
    train_ds = ds_split["train"]
    eval_ds = ds_split["test"]

    # Optionally, push the entire dataset (train + eval) to your HF account
    print("Pushing dataset to Hugging Face Hub...")
    # This will create a new dataset under your username or org
    dataset_repo = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
    ds_split.push_to_hub(dataset_repo,)


if __name__ == "__main__":
    main()
