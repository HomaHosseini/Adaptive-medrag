from src.medrag import MedRAG
from src.adapt_medrag import AdaptiveMedRAG
import json
import os
import requests
import re, ast

os.environ["ARVAN_API_KEY"] = "d60b8d53-5f6c-5eb6-828a-031d503221d7"
os.environ["ARVAN_ENDPOINT"] = "https://arvancloudai.ir/gateway/models/Llama-3.3-70B-Instruct-R1AI/wXwicG2w6RhESLlIg7KlsetpzCShAX1k0bXHGZLX9nDHLxr4mJwXDMW-2Vunju0LPAXGRBNA8Hg4NhbIgiU-6UwAscgx9QDw1vINQTl1636Buf89X2lEaj3jhmNVJ9Q5qjRgH_G-9CxWta0kJqLV_LtR3XQypA0QnGRFDXtMfKtcE4Q0w5zta-2-Ul-b1rMjyoCaLU2oe4h7NM8-23TJZsUjYp9JrcetdbTBdzh2TDjNK1Iy0I10_c_8RQIN0B9LAEWZihx5ela0-k8AEHGCTwvU5zqzRXjNpwAdEQ/v1"



# ============================================================
# MEDQA BENCHMARK
# ============================================================

import json
from datasets import load_dataset


# ------------------------------------------------------------
# 1. Parse model answer
# ------------------------------------------------------------
def parse_predicted_letter(raw_answer):
    """
    Extract A-F from JSON, Python-dict syntax, markdown,
    or natural-language answers.
    """

    if raw_answer is None:
        return "UNKNOWN"

    text = str(raw_answer).strip()

    # Remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    # --------------------------------------------------
    # 1. Try real JSON
    # --------------------------------------------------
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            value = str(parsed.get("answer", "")).strip().upper()

            match = re.match(r"^([A-F])(?:[\.\):\s]|$)", value)
            if match:
                return match.group(1)

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # --------------------------------------------------
    # 2. Try Python dict syntax:
    #    {'answer': 'A'}
    # --------------------------------------------------
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, dict):
            value = str(parsed.get("answer", "")).strip().upper()

            match = re.match(r"^([A-F])(?:[\.\):\s]|$)", value)
            if match:
                return match.group(1)

    except (ValueError, SyntaxError):
        pass

    # --------------------------------------------------
    # 3. Natural language:
    #    "The answer is A"
    # --------------------------------------------------
    match = re.search(
        r"\b(?:answer|correct answer)\s*(?:is|:)\s*([A-F])\b",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # --------------------------------------------------
    # 4. "A. Option text"
    # --------------------------------------------------
    match = re.search(
        r"(?m)^\s*([A-F])[\.\):]\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # --------------------------------------------------
    # 5. Just "A"
    # --------------------------------------------------
    match = re.fullmatch(r"\s*([A-F])\s*", text, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    return "UNKNOWN"

# ------------------------------------------------------------
# 2. Load MedQA
# ------------------------------------------------------------

def load_medqa_subset(num_samples=20):

    print(f"Loading {num_samples} MedQA questions...")

    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="test"
    )

    subset = dataset.select(range(num_samples))

    return [
        {
            "id": f"MedQA_{i+1}",
            "question": item["question"],
            "options": item["options"],
            "ground_truth": item["answer_idx"]
        }
        for i, item in enumerate(subset)
    ]


# ------------------------------------------------------------
# 3. Run Adaptive i-MedRAG
# ------------------------------------------------------------

def run_adaptive_benchmark(
    dataset,
    medrag_runner,
    max_iterations=3,
    n_queries=2,
    confidence_threshold=85
):

    results = []
    correct = 0
    total_iterations = 0

    print("\n" + "=" * 65)
    print("ADAPTIVE i-MedRAG")
    print("=" * 65)

    for item in dataset:

        print(f"\nEvaluating {item['id']}...")


        raw_answer,history,iterations,confidence,query_budget_history,total_queries  = medrag_runner.adaptive_answer(
            question=item["question"],
            options=item["options"],
            k=32,
            max_iterations=max_iterations,
            initial_queries=2,
            confidence_threshold=confidence_threshold
        )
        print("RAW FINAL ANSWER:")
        print(repr(raw_answer))
        print()
        predicted = parse_predicted_letter(raw_answer)
        is_correct = predicted == item["ground_truth"]

        if is_correct:
            correct += 1

        total_iterations += iterations

        result = {
            "id": item["id"],
            "predicted": predicted,
            "expected": item["ground_truth"],
            "correct": is_correct,
            "iterations": iterations,
            "confidence": confidence,
            "query_budget_history": query_budget_history,
            "total_queries": total_queries
        }

        results.append(result)

        print(
            f"  Iterations: {iterations} | "
            f"Queries: {total_queries} | "
            f"Budget history: {query_budget_history} | "
            f"Final confidence: {confidence:.1f}%"
        )

    accuracy = correct / len(dataset) * 100
    avg_iterations = total_iterations / len(dataset)

    return {
        "results": results,
        "accuracy": accuracy,
        "avg_iterations": avg_iterations,
        "total_iterations": total_iterations
    }


# ------------------------------------------------------------
# 4. Run Standard i-MedRAG
# ------------------------------------------------------------

def run_standard_benchmark(
    dataset,
    medrag_runner,
    fixed_rounds=3,
    n_queries=2
):

    results = []
    correct = 0

    print("\n" + "=" * 65)
    print("STANDARD i-MedRAG")
    print("=" * 65)

    for item in dataset:

        print(f"\nEvaluating {item['id']}...")

        raw_answer, history = medrag_runner.i_medrag_answer(
            question=item["question"],
            options=item["options"],
            k=32,
            n_rounds=fixed_rounds,
            n_queries=n_queries
        )
        print("RAW FINAL ANSWER:")
        print(repr(raw_answer))
        print()
        predicted = parse_predicted_letter(raw_answer)
        is_correct = predicted == item["ground_truth"]

        if is_correct:
            correct += 1

        result = {
            "id": item["id"],
            "predicted": predicted,
            "expected": item["ground_truth"],
            "correct": is_correct,
            "iterations": fixed_rounds
        }

        results.append(result)

        print(
            f"  Predicted: {predicted} | "
            f"Expected: {item['ground_truth']} | "
            f"Correct: {is_correct}"
        )

    accuracy = correct / len(dataset) * 100

    return {
        "results": results,
        "accuracy": accuracy,
        "avg_iterations": fixed_rounds,
        "total_iterations": fixed_rounds * len(dataset)
    }


# ------------------------------------------------------------
# 5. Create both models
# ------------------------------------------------------------

MODEL_NAME = "Arvan/Llama-3-3-70B-Instruct-R1AI"

adaptive_medrag = AdaptiveMedRAG(
    llm_name=MODEL_NAME,
    rag=True,
    follow_up=True,
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    corpus_cache=True
)

standard_medrag = MedRAG(
    llm_name=MODEL_NAME,
    rag=True,
    follow_up=True,
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    corpus_cache=True
)


# ------------------------------------------------------------
# 6. Load benchmark
# ------------------------------------------------------------

medqa_batch = load_medqa_subset(num_samples=2)


# ------------------------------------------------------------
# 7. Run experiments
# ------------------------------------------------------------

# standard = run_standard_benchmark(
#     dataset=medqa_batch,
#     medrag_runner=standard_medrag,
#     fixed_rounds=3,
#     n_queries=2
# )

adaptive = run_adaptive_benchmark(
    dataset=medqa_batch,
    medrag_runner=adaptive_medrag,
    max_iterations=3,
    n_queries=2,
    confidence_threshold=85
)


# ------------------------------------------------------------
# 8. Final comparison
# ------------------------------------------------------------

# iteration_savings = (
#     (standard["avg_iterations"] - adaptive["avg_iterations"])
#     / standard["avg_iterations"]
#     * 100
# )


# print("\n")
# print("=" * 75)
# print("FINAL COMPARISON")
# print("=" * 75)

# print(
#     f"{'Metric':<30}"
#     f"{'Standard':<20}"
#     f"{'Adaptive':<20}"
# )

# print("-" * 75)

# print(
#     f"{'Accuracy':<30}"
#     f"{standard['accuracy']:.2f}%{'':<14}"
#     f"{adaptive['accuracy']:.2f}%"
# )

# print(
#     f"{'Total iterations':<30}"
#     f"{standard['total_iterations']:<20}"
#     f"{adaptive['total_iterations']}"
# )

# print(
#     f"{'Average iterations/question':<30}"
#     f"{standard['avg_iterations']:<20.2f}"
#     f"{adaptive['avg_iterations']:.2f}"
# )

# print(
#     f"{'Iteration savings':<30}"
#     f"{'0.00%':<20}"
#     f"{iteration_savings:.2f}%"
# )

# print("=" * 75)


# # ------------------------------------------------------------
# # 9. Save results
# # ------------------------------------------------------------

# with open("standard_medrag_results.json", "w") as f:
#     json.dump(standard["results"], f, indent=4)

# with open("adaptive_medrag_results.json", "w") as f:
#     json.dump(adaptive["results"], f, indent=4)

# print("\nResults saved.")