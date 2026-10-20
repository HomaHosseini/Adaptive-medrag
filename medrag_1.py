
import time
import json
import re
import ast

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

from src.medrag import MedRAG



load_dotenv()

NUM_SAMPLES = 50
START_INDEX = 550
SEED = 42

MODEL_NAME = "ArvanCloud/GPT-4o-mini"

RETRIEVER_NAME = "MedCPT"
CORPUS_NAME = "Textbooks"

K = 10

# Standard i-MedRAG: one iteration, two queries
STANDARD_ROUNDS = 1
STANDARD_QUERIES_PER_ROUND = 2

OUTPUT_FILE = "experiment_results_1iter_550_600.json"


def parse_predicted_letter(raw_answer):
    """Extract the selected answer letter A-F."""

    if raw_answer is None:
        return "UNKNOWN"

    text = str(raw_answer).strip()

    # Remove Markdown code fences
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # Try JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            value = str(parsed.get("answer", "")).strip().upper()

            match = re.match(
                r"^([A-F])(?:[.)\s:]|$)",
                value
            )
            if match:
                return match.group(1)

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try Python dictionary
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, dict):
            value = str(parsed.get("answer", "")).strip().upper()

            match = re.match(
                r"^([A-F])(?:[.)\s:]|$)",
                value
            )
            if match:
                return match.group(1)

    except (ValueError, SyntaxError):
        pass

    # Natural language: "The answer is C"
    match = re.search(
        r"\b(?:answer|correct answer)\s*(?:is|:)\s*([A-F])\b",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Answer on its own line: "C." or "C)"
    match = re.search(
        r"(?m)^\s*([A-F])[.):]\s*",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Response contains only a letter
    match = re.fullmatch(r"\s*([A-F])\s*", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "UNKNOWN"




def load_medqa_subset(num_samples=1):

    print(f"Loading {num_samples} MedQA question(s)...")

    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="train"
    )

    dataset = dataset.shuffle(seed=SEED)

    end_index = START_INDEX + num_samples 

    if end_index > len(dataset):
        raise ValueError(
            f"Requested indices {START_INDEX}:{end_index}, "
            f"but the dataset contains only {len(dataset)} questions."
        )

    subset = dataset.select(
        range(START_INDEX, end_index)
    )

    questions = []

    for i, item in enumerate(subset):

        questions.append({
            "id": f"MedQA_{START_INDEX + i + 1}",
            "question": item["question"],
            "options": item["options"],
            "ground_truth": item["answer_idx"]
        })

    return questions




def run_standard_benchmark(
    dataset,
    medrag_runner,
    fixed_rounds=1,
    n_queries=2,
    k=10
):

    results = []

    correct = 0
    total_iterations = 0
    total_queries = 0

    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    total_latency = 0.0

    print("\n")
    print("=" * 100)
    print("STANDARD i-MedRAG — ONE ITERATION")
    print("=" * 100)

    for item in dataset:

        print("\n")
        print("-" * 100)
        print(f"Evaluating {item['id']} | STANDARD")
        print("-" * 100)

        question_start = time.perf_counter()

        try:

            raw_answer, history, metrics = (
                medrag_runner.i_medrag_answer(
                    question=item["question"],
                    options=item["options"],
                    k=k,
                    n_rounds=fixed_rounds,
                    n_queries=n_queries
                )
            )

            latency_seconds = (
                time.perf_counter() - question_start
            )

            predicted = parse_predicted_letter(raw_answer)

            is_correct = (
                predicted == item["ground_truth"]
            )

            if is_correct:
                correct += 1

            # Read metrics returned by the MedRAG implementation.
            llm_calls = metrics.get("llm_calls", 0)
            prompt_tokens = metrics.get("prompt_tokens", 0)
            completion_tokens = metrics.get("completion_tokens", 0)
            question_tokens = metrics.get("total_tokens", 0)

            # Configured counts, not independently verified retrievals.
            question_queries = fixed_rounds * n_queries

            total_iterations += fixed_rounds
            total_queries += question_queries

            total_llm_calls += llm_calls
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += question_tokens
            total_latency += latency_seconds

            result = {
                "id": item["id"],
                "configuration": "standard",

                "question": item["question"],
                "options": item["options"],
                "expected": item["ground_truth"],
                "predicted": predicted,
                "correct": is_correct,

                "raw_answer": raw_answer,

                "iterations": fixed_rounds,
                "queries_per_iteration": n_queries,
                "queries_generated": question_queries,

                "llm_calls": llm_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": question_tokens,

                "latency_seconds": latency_seconds,
                "history": history
            }

        except Exception as e:

            latency_seconds = (
                time.perf_counter() - question_start
            )

            print(f"ERROR evaluating {item['id']}: {e}")

            result = {
                "id": item["id"],
                "configuration": "standard",

                "question": item["question"],
                "options": item["options"],
                "expected": item["ground_truth"],
                "predicted": "ERROR",
                "correct": False,

                "raw_answer": None,
                "error": str(e),

                "iterations": 0,
                "queries_per_iteration": n_queries,
                "queries_generated": 0,

                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,

                "latency_seconds": latency_seconds,
                "history": []
            }

            total_latency += latency_seconds

        results.append(result)

        print(f"Predicted: {result['predicted']}")
        print(f"Expected: {result['expected']}")
        print(f"Correct: {result['correct']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Queries generated: {result['queries_generated']}")
        print(f"LLM calls: {result['llm_calls']}")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Latency: {result['latency_seconds']:.2f}s")


    n = len(dataset)
    successful_answers = sum(
        result["predicted"] != "ERROR"
        for result in results
    )

    summary = {
        "configuration": "standard",

        "accuracy": (
            correct / n * 100 if n > 0 else 0.0
        ),
        "correct": correct,
        "total_questions": n,
        "successful_answers": successful_answers,
        "failed_answers": n - successful_answers,

        "avg_iterations": (
            total_iterations / n if n > 0 else 0.0
        ),
        "total_iterations": total_iterations,

        "avg_queries": (
            total_queries / n if n > 0 else 0.0
        ),
        "total_queries": total_queries,

        "avg_llm_calls": (
            total_llm_calls / n if n > 0 else 0.0
        ),
        "total_llm_calls": total_llm_calls,

        "avg_prompt_tokens": (
            total_prompt_tokens / n if n > 0 else 0.0
        ),
        "total_prompt_tokens": total_prompt_tokens,

        "avg_completion_tokens": (
            total_completion_tokens / n if n > 0 else 0.0
        ),
        "total_completion_tokens": total_completion_tokens,

        "avg_total_tokens": (
            total_tokens / n if n > 0 else 0.0
        ),
        "total_tokens": total_tokens,

        "avg_latency_seconds": (
            total_latency / n if n > 0 else 0.0
        ),
        "total_latency_seconds": total_latency
    }

    print("\n")
    print("=" * 100)
    print("STANDARD i-MedRAG SUMMARY")
    print("=" * 100)

    print(f"Accuracy: {summary['accuracy']:.2f}% "
          f"({correct}/{n})")
    print(f"Successful answers: {successful_answers}/{n}")
    print(f"Average iterations: {summary['avg_iterations']:.2f}")
    print(f"Average queries: {summary['avg_queries']:.2f}")
    print(f"Average LLM calls: {summary['avg_llm_calls']:.2f}")
    print(f"Average prompt tokens: {summary['avg_prompt_tokens']:.2f}")
    print(f"Average completion tokens: {summary['avg_completion_tokens']:.2f}")
    print(f"Average total tokens: {summary['avg_total_tokens']:.2f}")
    print(f"Average latency: {summary['avg_latency_seconds']:.2f}s")

    print("=" * 100)

    return {
        **summary,
        "results": results
    }



def save_json(data, filepath):

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )



if __name__ == "__main__":

    # Load questions
    medqa_batch = load_medqa_subset(
        num_samples=NUM_SAMPLES
    )

    print("\n")
    print("=" * 100)
    print("QUESTIONS USED")
    print("=" * 100)

    for item in medqa_batch:
        print(
            f"{item['id']} -> "
            f"Ground truth: {item['ground_truth']}"
        )

    # Initialize Standard MedRAG only
    print("\nInitializing STANDARD MedRAG...")

    standard_medrag = MedRAG(
        llm_name=MODEL_NAME,
        rag=True,
        follow_up=True,
        retriever_name=RETRIEVER_NAME,
        corpus_name=CORPUS_NAME,
        corpus_cache=True
    )

    # Run one standard iteration
    result = run_standard_benchmark(
        dataset=medqa_batch,
        medrag_runner=standard_medrag,
        fixed_rounds=STANDARD_ROUNDS,
        n_queries=STANDARD_QUERIES_PER_ROUND,
        k=K
    )

    # Prepare output
    output = {
        "experiment_settings": {
            "num_samples": NUM_SAMPLES,
            "start_index": START_INDEX,
            "model": MODEL_NAME,
            "retriever": RETRIEVER_NAME,
            "corpus": CORPUS_NAME,
            "k": K,
            "seed": SEED,
            "standard_rounds": STANDARD_ROUNDS,
            "standard_queries_per_round": STANDARD_QUERIES_PER_ROUND
        },

        "questions": medqa_batch,

        "summary": {
            key: value
            for key, value in result.items()
            if key != "results"
        },

        "results": result["results"]
    }

    save_json(output, OUTPUT_FILE)

    print("\n")
    print("=" * 100)
    print("EXPERIMENT COMPLETED")
    print("=" * 100)

    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Questions: {result['total_questions']}")
    print(f"Average iterations: {result['avg_iterations']:.2f}")
    print(f"Average queries: {result['avg_queries']:.2f}")
    print(f"Average latency: {result['avg_latency_seconds']:.2f}s")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("=" * 100)