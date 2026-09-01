from typing import List, Dict
import pandas as pd
from src.retrieval.retriever import Retriever


BUILTIN_TEST_SET = [
    {"query": "What is supervised learning and how does it differ from unsupervised learning?",
     "relevant_sources": ["machine_learning.txt"]},
    {"query": "How does gradient descent optimize neural network weights?",
     "relevant_sources": ["machine_learning.txt"]},
    {"query": "What are indexes in a relational database and why do they speed up queries?",
     "relevant_sources": ["databases.txt"]},
    {"query": "Explain ACID properties in database transactions",
     "relevant_sources": ["databases.txt"]},
    {"query": "How does Python handle memory management and garbage collection?",
     "relevant_sources": ["python_programming.txt"]},
    {"query": "What are Python decorators and how are they used?",
     "relevant_sources": ["python_programming.txt"]},
    {"query": "What is the difference between ETL and ELT pipelines?",
     "relevant_sources": ["data_engineering.txt"]},
    {"query": "How do columnar storage formats like Parquet improve query performance?",
     "relevant_sources": ["data_engineering.txt"]},
    {"query": "What is the shared responsibility model in cloud computing?",
     "relevant_sources": ["cloud_computing.txt"]},
    {"query": "How do SQL window functions like ROW_NUMBER and RANK work?",
     "relevant_sources": ["sql_fundamentals.txt"]},
]


def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    hits = sum(1 for s in top_k if s in relevant_sources)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    hits = sum(1 for s in top_k if s in relevant_sources)
    return hits / len(relevant_sources) if relevant_sources else 0.0


def evaluate(retriever: Retriever, test_set: List[Dict] = None,
             k_values: List[int] = None) -> pd.DataFrame:
    if test_set is None:
        test_set = BUILTIN_TEST_SET
    if k_values is None:
        k_values = [1, 3, 5, 10]

    max_k = max(k_values)
    rows = []

    for item in test_set:
        results = retriever.query(item["query"], top_k=max_k)
        retrieved = [r.chunk.source for r in results]
        row = {"query": item["query"][:60]}
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(retrieved, item["relevant_sources"], k)
            row[f"recall@{k}"] = recall_at_k(retrieved, item["relevant_sources"], k)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def mean_metrics(df: pd.DataFrame, k_values: List[int] = None) -> pd.DataFrame:
    if k_values is None:
        k_values = [1, 3, 5, 10]
    records = []
    for k in k_values:
        p_col = f"precision@{k}"
        r_col = f"recall@{k}"
        if p_col in df.columns and r_col in df.columns:
            records.append({
                "k": k,
                "mean_precision": df[p_col].mean(),
                "mean_recall": df[r_col].mean(),
            })
    return pd.DataFrame(records)
