# Medical RAG Chatbot (Symptoms → Possible Diseases)

A retrieval-augmented chatbot that suggests **possible diseases** based on user-reported symptoms using **open-source large language models**.

The system retrieves relevant disease/symptom records from a small knowledge base and then uses an LLM to generate a concise response.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not** medical advice, diagnosis, or treatment.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Symptom normalization & preprocessing
- Semantic search using SentenceTransformers
- LLM inference with **BioMistral 7B (4-bit)**
- GPU-friendly (optimized for Kaggle T4)
- CLI-based chatbot interface
- Optional TF-IDF baseline
- Fine-tuning pipeline included in notebook (LoRA / QLoRA)

---

## Architecture

```

User Symptoms
↓
Embedding Model (SentenceTransformers)
↓
Semantic Search (Top-K contexts)
↓
Prompt Construction
↓
LLM Generation (BioMistral)
↓
Response + Disclaimer

```

---

## Example

**Input**
```

fever, cough, sore throat, body aches

```

**Output (example)**
```

* Influenza
* Common cold

This is not a medical diagnosis. Please consult a doctor.

```

---

## Tech Stack

- Python
- Hugging Face Transformers
- SentenceTransformers
- BitsAndBytes (4-bit quantization)
- PyTorch
- scikit-learn
- Pandas
- PEFT (LoRA / QLoRA)
- Datasets

---

## Repository Structure

```

medical-rag-chatbot/
│
├── medical_rag_chatbot.py
├── README.md
├── requirements.txt
├── RUN.md
├── notebooks/
│   └── medical_chatbot_completed.ipynb
├── src/
└── data/

```

---

## Dataset Source (Kaggle)

This project uses the **Disease Symptom Prediction** dataset from Kaggle:

https://www.kaggle.com/datasets/karthikudyawar/disease-symptom-prediction

Only the file **`dataset.csv`** is used.

The dataset contains:

- `Disease` column  
- Multiple `Symptom_*` columns describing associated symptoms  

---

## How to Add the Dataset in Kaggle

1. Open your Kaggle notebook  
2. Click **Input** (right sidebar)  
3. Click **+ Add Input**  
4. Search for:

```

disease symptom prediction

```

or paste the dataset URL above

5. Click **Add**
6. The dataset will be available at:

```

/kaggle/input/disease-symptom-prediction/dataset.csv

````

---

## Dataset Format (After Processing)

The pipeline automatically:

- Replaces underscores with spaces in symptom names  
- Merges `Symptom_*` columns into a single `Symptoms` column  
- Removes duplicate symptom lists  
- Groups entries by disease  

Final required columns:

- `Disease`
- `Symptoms`

---

## Quickstart (Kaggle – GPU T4)

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

or

```bash
pip install -U transformers accelerate sentencepiece bitsandbytes optimum sentence-transformers peft datasets
```

---

### 2. Set dataset path

```bash
export SYMPTOM_DATASET=/kaggle/input/disease-symptom-prediction/dataset.csv
```

---

### 3. Run

```bash
python medical_rag_chatbot.py
```

---

## Quickstart (Local)

### Install

```bash
pip install -U transformers accelerate sentencepiece bitsandbytes optimum sentence-transformers peft datasets torch scikit-learn pandas
```

### Run

```bash
export SYMPTOM_DATASET=/path/to/your_dataset.csv
python medical_rag_chatbot.py
```

---

## Model Choice

This project defaults to:

> **BioMistral 7B (4-bit)** via `bitsandbytes`

Reasons:

* Fast inference on T4 GPUs
* Stable on Kaggle
* Medical-domain tuned
* Avoids fragile GPTQ builds

---

## Safety Notes

This chatbot:

* Can hallucinate
* Does not verify against clinical guidelines
* Does not detect emergencies
* Does not provide dosages

For production use, add:

* Emergency symptom detection
* Verified sources (CDC / NIH / PubMed)
* Citation system
* Rule-based guardrails
* Monitoring & evaluation

---

## Academic / Portfolio Use

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Embedding-based semantic search
* LLM inference optimization
* Quantization techniques
* Medical NLP preprocessing
* Applied ML system design
