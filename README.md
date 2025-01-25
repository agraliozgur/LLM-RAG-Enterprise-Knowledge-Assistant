# Enterprise-Knowledge-Assistant

A **Retrieval-Augmented Generation (RAG)** system for answering corporate document queries using an LLM, with alignment for security and ethics, plus MLOps integrations.

---

## Overview

**Enterprise Knowledge Assistant** is designed to:
- Preprocess corporate documents from various formats.
- Embed and index documents in a vector database (e.g., Qdrant).
- Leverage an LLM (default: `google/flan-t5-large`) for generating answers.
- Use a retrieval pipeline to provide grounded responses.
- Apply alignment checks to ensure compliance with corporate policies.

Key components:

- **Synthetic Data Generation**: `create_synthetic_data_gemini.py`  
- **Data Preprocessing**: `scripts/data_preprocessing.py`  
- **Utilities**: `scripts/utils.py`  
- **Project Settings**: `config/project_settings.yaml`  
- **Embedding & Indexing**: `scripts/embed_and_index.py`  
- **Alignment & Policy**: `scripts/alignment.py`  
- **Comparison**: `scripts/compare_llm_vs_rag.py` (LLM-only vs. RAG)  
- **MLOps Pipeline**: `scripts/mlops_pipeline.py`  

---

## Project Structure


corporate_info_assistant/
├── config/
│   └── project_settings.yaml
├── data/
│   └── raw/                
      └── image_files/
      ├── txt_files/
      ├── pdf_files/
      ├── html_files/
│   └── cleaned_data/  
      └───all_processed_data.jsonl 
│   └── vector_index/       
├── scripts/
│   ├── data_preprocessing.py
│   ├── create_synthetic_data_gemini.py
│   ├── save_dataset_to_huggingface.py
│   ├── embed_and_index.py
│   ├── alignment.py
│   ├── rag_inference.py
│   ├── compare_llm_vs_rag.py
│   ├── mlops_pipeline.py
│   └── utils.py
├── tests/
│   └── test_alignment.py            # model test differences and example of questions
├── requirements.txt
├── .gitignore
└── README.md
docker run --name qdrant -v ./data/vector_index:/qdrant/storage  -p 6333:6333 qdrant/qdrant:latest


---

## Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/corporate_info_assistant.git
   cd corporate_info_assistant
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv rag-llm-env
   source rag-llm-env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   or
   pip install -r requirements.txt --break-system-packages 
   ```
   Make sure `requirements.txt` includes:
   - `torch`, `transformers`
   - `langchain`, `qdrant-client`
   - `pydantic`, `pyyaml`
   - `sentence-transformers` (or any embedding model dependencies)
   - `numpy`, `pandas` (if needed)
   - etc.

4. **Configure Qdrant** (if using local Docker):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
   Or use your hosted Qdrant instance.
   ```bash
   docker run --name qdrant -v ./data/vector_index:/qdrant/storage  -p 6333:6333 qdrant/qdrant:latest
    ```
5. **Adjust `project_settings.yaml`** in `/config/` to match your environment.

---

## Usage

### 1. Creating synthetic data using Gemini1.5-Flash and publishing data to HuggingFace as a dataset
```bash
python scripts/create_synhetic_data_gemini.py
python scripts/save_dataset_to_huggingface.py 
```
- Generates additional corporate data for testing or training to txt file and publish to huggignface as a dataset.`data/cleaned_data/`.
### 2. Preprocess Data
```bash
python scripts/data_preprocessing.py
```
- Cleans, tokenizes, or chunks raw files into `data/cleaned_data/`. These can read and get text from  ".pdf", ".docx", ".html", ".txt", ".rtf", ".pptx", ".xlsx", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",".zip", ".tar", ".gz", and ".bz2" extension data.

### 2. Create Synthetic Test Data (Optional)
```bash
python scripts/create_synthetic_data_gemini.py
```
- 

### 3. Embed & Index and Testing Similarity
```bash
python scripts/embed_and_index.py
python scripts/get_similar_chunks_qdrant.py
```
- Converts cleaned text chunks to embeddings and stores them in Qdrant. Also can test with questions to find most similartiy of this model in the dataset.

### 4. Compare LLM vs. RAG
```bash
python scripts/compare_llm_vs_rag.py
```
- Enters an interactive loop.  
- You can type queries and see the difference between direct LLM answers and RAG-based answers (with alignment).

### 5. Run MLOps Pipeline
```bash
python scripts/mlops_pipeline.py
```
- Placeholder for continuous monitoring, triggers, CI/CD, etc.

---

## Saving & Reusing a Fine-Tuned Model

If you fine-tune `google/flan-t5-large`:
1. Run your fine-tuning script (not shown here).
2. Use `utils.save_model(finetuned_model, tokenizer, "./models/my_finetuned_flan_t5_large")`.
3. Update `project_settings.yaml`:
   ```yaml
   models:
     llm_model_name: "./models/my_finetuned_flan_t5_large"
   ```
4. Future scripts will saving & reusing a Fine-Tuned model and load your local checkpoint instead of the base model from Hugging Face.

---

## Troubleshooting

- **Connection Errors**: Verify Qdrant is running (`docker ps`) and `project_settings.yaml` has the correct URL.  
- **Model Loading Issues**: Check you have enough GPU/CPU resources, or reduce model size.  
- **Alignment or Policy**: If `alignment.py` missing, create or stub it out.  

---

## Contributing

1. Fork the repo and create a new branch for your feature.  
2. Submit a PR with detailed explanations and tests.  
3. Ensure code follows Pythonic style and is well-documented.

---

## License

Specify your license (e.g., MIT, Apache 2.0) or keep it proprietary within your organization.

---
