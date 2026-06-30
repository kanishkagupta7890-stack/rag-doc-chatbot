"""
evaluate.py
------------
This script EVALUATES our RAG chatbot using RAGAS (RAG Assessment) —
an industry-standard evaluation framework. This is what separates a
"tutorial project" from a "production-aware project" on your resume.

WHY EVALUATE A RAG SYSTEM?
Without evaluation, you have no idea if your chatbot is actually good.
RAGAS gives us 4 concrete metrics:

1. FAITHFULNESS (0 to 1)
   Does the answer only contain claims that are actually supported by the
   retrieved context? Measures HALLUCINATION. High score = the LLM isn't
   making things up.

2. ANSWER RELEVANCY (0 to 1)
   Is the answer actually relevant to the question asked? An answer can be
   100% faithful to the context but still dodge the actual question.

3. CONTEXT PRECISION (0 to 1)
   Of the chunks we retrieved, how many were actually relevant/useful?
   Measures whether our RETRIEVAL step is pulling in noise.

4. CONTEXT RECALL (0 to 1)
   Did we retrieve ALL the information needed to answer the question
   correctly? Low recall = we're missing important chunks (chunk_size or
   k might need tuning).

HOW TO RUN:
    python evaluate.py

This will:
  1. Load your sample document
  2. Run a set of test questions through your RAG pipeline
  3. Score each answer using RAGAS
  4. Print a results table + save to evaluation_results.csv
"""

import os
import pandas as pd
from dotenv import load_dotenv

from rag_pipeline import process_document_and_create_chain
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

TEST_QUESTIONS = [
    {
        "question": "How many vacation days do employees get per year?",
        "ground_truth": "Full-time employees accrue 1.5 days of paid vacation per month, totaling 18 days per year.",
    },
    {
        "question": "How many days per week can employees work remotely?",
        "ground_truth": "Employees may work remotely up to 3 days per week, subject to manager approval.",
    },
    {
        "question": "How far in advance must vacation requests be submitted?",
        "ground_truth": "Vacation requests must be submitted at least 2 weeks in advance for approval.",
    },
    {
        "question": "What is the receipt requirement for expense reimbursement?",
        "ground_truth": "Receipts are required for any expense over $25.",
    },
    {
        "question": "How often are performance reviews conducted?",
        "ground_truth": "Performance reviews are conducted twice a year, in June and December.",
    },
]


def run_evaluation(document_path: str = "sample_docs/company_policy.txt"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Set GROQ_API_KEY in your .env file before running evaluation.")

    print(f"Loading and processing document: {document_path}")
    qa_chain, num_chunks = process_document_and_create_chain(document_path, groq_api_key)
    print(f"Document split into {num_chunks} chunks.\n")

    questions, ground_truths, answers, contexts = [], [], [], []

    for item in TEST_QUESTIONS:
        print(f"Asking: {item['question']}")
        result = qa_chain.invoke({"query": item["question"]})

        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])
        answers.append(result["result"])
        contexts.append([doc.page_content for doc in result["source_documents"]])

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    judge_llm = LangchainLLMWrapper(
        ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    print("\nRunning RAGAS evaluation (this calls the LLM multiple times, may take a minute)...")
    # max_workers=1 and increased timeout reduce Groq free-tier rate-limit
    # failures, which otherwise show up as NaN scores in the results.
    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=120, max_retries=5, max_wait=30, max_workers=1)

    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    results_df = results.to_pandas()

    # NOTE: Different ragas versions name the question column differently
    # ("question" in older versions, "user_input" in 0.2.x+). We detect
    # whichever one is present so this script works across versions.
    question_col = "question" if "question" in results_df.columns else "user_input"

    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                   if c in results_df.columns]

    print("\n=== Per-Question Scores ===")
    print(results_df[[question_col] + metric_cols].to_string(index=False))

    print("\n=== Average Scores (put these on your resume!) ===")
    for metric in metric_cols:
        avg = results_df[metric].mean()
        print(f"  {metric:20s}: {avg:.2%}")

    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nFull results saved to evaluation_results.csv")

    return results_df


if __name__ == "__main__":
    run_evaluation()
