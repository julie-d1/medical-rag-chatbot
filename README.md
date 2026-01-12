# Medical RAG Chatbot (Symptoms → Possible Diseases)

A retrieval-augmented chatbot that suggests **possible diseases** based on user-reported symptoms using **open-source LLMs**.
It retrieves relevant disease/symptom “medical records” from a small knowledge base, then prompts a model to produce a concise answer.

> **Disclaimer:** This project is for educational and research purposes only. It is **not** medical advice, diagnosis, or treatment.

## What’s inside

- **RAG pipeline**
  - Symptom dataset → normalized symptom strings
  - Embeddings with **SentenceTransformers** (`all-MiniLM-L6-v2`)
  - Retrieval with `semantic_search` (top-k contexts)
  - Generation with an LLM using retrieved contexts

- **Recommended model for Kaggle T4**
  - **BioMistral 7B** in **4-bit** via `bitsandbytes` (fast + stable on Kaggle)

- **Notebook source**
  - `medical_chatbot_completed.ipynb` (completed notebook)

## Architecture

User Symptoms → Embedding Search → Retrieve Context → LLM Prompt → Response (+ Disclaimer)

## Example

**Input**
```text
fever, cough, sore throat, body aches
```

**Output (example)**
- Influenza  
- Common cold  

_Disclaimer: This is not a medical diagnosis. Consult a licensed physician._

## Quickstart (Kaggle / GPU)

1. Add a symptom dataset (CSV) to your Kaggle notebook (or use one in `/kaggle/input/...`).
2. Install dependencies:
```bash
pip install -U transformers accelerate sentencepiece bitsandbytes optimum sentence-transformers peft datasets
```
3. Run the script by setting the dataset path:
```bash
SYMPTOM_DATASET=/kaggle/input/<dataset>/<file>.csv python medical_rag_chatbot.py
```

## Quickstart (Local)

Install:
```bash
pip install -U transformers accelerate sentencepiece bitsandbytes optimum sentence-transformers peft datasets torch scikit-learn pandas
```

Run:
```bash
export SYMPTOM_DATASET=/path/to/your_dataset.csv
python medical_rag_chatbot.py
```

## Dataset format

Expected columns:
- `Disease`
- either:
  - multiple `Symptom*` columns (e.g., `Symptom_1`, `Symptom_2`, …), **or**
  - a single `Symptoms` column

The script will:
- replace underscores with spaces
- merge symptom columns into `Symptoms`
- remove duplicate symptom lists
- group by disease

## Notes on safety

This chatbot can hallucinate and should not be used for real medical decisions.
For a production-grade system, add:
- emergency escalation (e.g., chest pain, stroke symptoms)
- refusal rules for prescriptions/dosage
- citations to verified sources (CDC/NIH/PubMed)
- evaluation + monitoring
