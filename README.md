# Enterprise-Knowledge-Assistant

1. **compare_llm_vs_rag.py** (revised / final version)  
2. **project_settings.yaml** (sample YAML configuration)  
3. **utils.py** (common utilities for logging, config loading, device detection, embedding retrieval, etc.)  
4. **mlops_pipeline.py** (a simple placeholder for MLOps tasks)  
5. **README.md** (detailed instructions for setup, running, and extending the project)

This example demonstrates:

- How to **centralize configurations** in `project_settings.yaml`.  
- How to **load configurations** in each script using `utils.py`.  
- How to **compare direct LLM vs. RAG** in `compare_llm_vs_rag.py` (your core logic).  
- How to **include alignment** logic for final answers.  
- Where and how to **save fine-tuned models** (best practices).  
- Basic **MLOps pipeline** structure in `mlops_pipeline.py`.

Below is a sample **README** that explains how to set up, run, and maintain the project. Adjust details to match your environment and organization standards.

```markdown

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

```
corporate_info_assistant/
├── config/
│   └── project_settings.yaml        # Global YAML config
├── data/
│   ├── raw/                         # Unprocessed files
│   ├── cleaned_data/                # Preprocessed data (JSONL, logs, etc.)
│   └── vector_index/                # Local Qdrant or 
│   ├── cleaned_data/                # Local Qdrant or other index store
├── scripts/
│   ├── data_preprocessing.py
│   ├── create_synthetic_data_gemini.py
│   ├── embed_and_index.py
│   ├── alignment.py
│   ├── rag_inference.py
│   ├── compare_llm_vs_rag.py
│   ├── mlops_pipeline.py
│   └── utils.py
├── models/
│   └── my_finetuned_flan_t5_large/  # Example local model checkpoint
├── tests/                           # Unit/Integration tests
├── requirements.txt
└── README.md
```

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

### 1. Preprocess Data
```bash
python scripts/data_preprocessing.py
```
- Cleans, tokenizes, or chunks raw files into `data/cleaned_data/`.

### 2. Create Synthetic Test Data (Optional)
```bash
python scripts/create_synthetic_data_gemini.py
```
- Generates additional corporate data for testing or training.

### 3. Embed & Index
```bash
python scripts/embed_and_index.py
```
- Converts cleaned text chunks to embeddings and stores them in Qdrant.

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
4. Future scripts will load your local checkpoint instead of the base model from Hugging Face.

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

**Happy Coding!**  
_Enterprise Knowledge Assistant_  
```

---

# Final Notes

- The above **sample code** and **config** files are designed to be **modular** and **extensible**.  
- Adjust imports and references (`langchain.embeddings`, `langchain.vectorstores`, `alignment`) to match your actual project structure.  
- Ensure your environment (`requirements.txt`) is up to date with all necessary packages.  
- Remember to **test thoroughly** (both unit tests in `tests/` and integration tests with real data).  

With this structure, you have:
- A clear place for **project-wide configuration** (`project_settings.yaml`)  
- Centralized **utility functions** in `utils.py`  
- A straightforward script to **compare direct LLM vs. RAG** with alignment in `compare_llm_vs_rag.py`  
- A scaffolding for **MLOps** monitoring and future integration in `mlops_pipeline.py`  
- A well-documented **README** for other developers and users.  

Use this as a **foundation** to build a robust, production-ready **Enterprise Knowledge Assistant** system. Good luck!