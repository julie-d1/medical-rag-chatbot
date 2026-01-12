"""
Medical RAG Chatbot
------------------
A retrieval-augmented chatbot that suggests possible diseases based on user-reported symptoms.

DISCLAIMER: Educational use only. Not medical advice or diagnosis.
"""

from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer, util

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# -----------------------------
# Data utilities
# -----------------------------

def load_symptom_dataset(csv_path: str) -> pd.DataFrame:
    '''
    Load a symptom→disease dataset.

    Expected columns:
      - 'Disease'
      - multiple 'Symptom_*' columns OR a single 'Symptoms' column
    '''
    return pd.read_csv(csv_path)


def normalize_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    '''
    - Replace underscores in symptom text with spaces
    - Combine Symptom_* columns into a single 'Symptoms' column if needed
    - Drop duplicates by symptom list
    - Group by disease to create one combined symptom string per disease
    '''
    symptom_cols = [c for c in df.columns if str(c).startswith("Symptom")]
    if symptom_cols:
        df[symptom_cols] = df[symptom_cols].replace("_", " ", regex=True)
        df["Symptoms"] = df[symptom_cols].apply(
            lambda row: ", ".join([s for s in row if pd.notnull(s)]),
            axis=1,
        )
        df.drop(symptom_cols, axis=1, inplace=True)

    if "Symptoms" not in df.columns or "Disease" not in df.columns:
        raise ValueError("Dataset must contain 'Disease' and 'Symptoms' (or Symptom_* columns).")

    df = df.dropna(subset=["Disease", "Symptoms"]).reset_index(drop=True)

    # Drop duplicates based on Symptoms (symptom-list duplicates)
    df = df.drop_duplicates(subset=["Symptoms"]).reset_index(drop=True)

    # Group by disease: one row per disease
    df = df.groupby("Disease")["Symptoms"].apply(lambda x: ", ".join(x)).reset_index()

    return df


def build_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df["Symptoms"])
    return vectorizer, matrix


# -----------------------------
# RAG (Embeddings + Retrieval)
# -----------------------------

def build_corpus(df: pd.DataFrame) -> List[str]:
    df = df.copy()
    df["Text"] = df.apply(lambda row: f"Disease: {row['Disease']}. Symptoms: {row['Symptoms']}", axis=1)
    return df["Text"].tolist()


def embed_corpus(corpus: List[str], model_name: str = "all-MiniLM-L6-v2"):
    embed_model = SentenceTransformer(model_name)
    corpus_embeddings = embed_model.encode(corpus, convert_to_tensor=True)
    return embed_model, corpus_embeddings


def retrieve_contexts(
    user_input: str,
    corpus: List[str],
    embed_model: SentenceTransformer,
    corpus_embeddings,
    top_k: int = 2,
) -> List[str]:
    query_embedding = embed_model.encode(user_input, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    return [corpus[h["corpus_id"]] for h in hits]


# -----------------------------
# BioMistral (recommended for Kaggle) - 4-bit inference
# -----------------------------

def load_biomistral_4bit(model_name: str = "BioMistral/BioMistral-7B"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, prompt: str, max_new_tokens: int = 250) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


def rag_prompt(contexts: List[str], user_input: str) -> str:
    return (
        "You are a medical assistant. Based on the medical records below, "
        "suggest the top 2 possible diseases the user might have. Be concise and use bullet points.\n\n"
        "Include a disclaimer at the end: this is not a medical diagnosis and the user should consult a doctor.\n\n"
        "Medical Records:\n"
        + "\n".join(contexts)
        + f"\n\nUser Symptoms: {user_input}\n\nYour Response:"
    )


def biomistral_rag_response(
    user_input: str,
    corpus: List[str],
    embed_model: SentenceTransformer,
    corpus_embeddings,
    tokenizer,
    model,
) -> str:
    contexts = retrieve_contexts(user_input, corpus, embed_model, corpus_embeddings, top_k=2)
    prompt = rag_prompt(contexts, user_input)
    return generate_response(tokenizer, model, prompt)


def chatbot_loop_biomistral(
    corpus: List[str],
    embed_model: SentenceTransformer,
    corpus_embeddings,
    tokenizer,
    model,
):
    print("ChatBot: I can help suggest possible diseases based on your symptoms.")
    print("Type your symptoms ('fever, cough, sore throat'), or type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("ChatBot: Goodbye!")
            print("Note: This is not a medical diagnosis. Always consult a licensed physician.")
            break

        response = biomistral_rag_response(
            user_input=user_input,
            corpus=corpus,
            embed_model=embed_model,
            corpus_embeddings=corpus_embeddings,
            tokenizer=tokenizer,
            model=model,
        )
        print(f"ChatBot: {response}\n")
        print("Note: This is not a medical diagnosis. Always consult a licensed physician.\n")


def main():
    csv_path = os.environ.get("SYMPTOM_DATASET", "").strip()
    if not csv_path:
        print("ERROR: Please set the dataset path via environment variable SYMPTOM_DATASET.")
        print("Example: SYMPTOM_DATASET=/kaggle/input/<dataset>/<file>.csv python medical_rag_chatbot.py")
        sys.exit(1)

    df_raw = load_symptom_dataset(csv_path)
    df = normalize_symptoms(df_raw)

    # Optional TF-IDF build (kept for parity with the notebook)
    _vectorizer, _tfidf_matrix = build_tfidf(df)

    corpus = build_corpus(df)
    embed_model, corpus_embeddings = embed_corpus(corpus)

    tokenizer, model = load_biomistral_4bit()
    chatbot_loop_biomistral(corpus, embed_model, corpus_embeddings, tokenizer, model)


if __name__ == "__main__":
    main()
