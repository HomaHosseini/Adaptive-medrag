import time
import json
import re
import ast
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest

from datasets import load_dataset
from dotenv import load_dotenv

from src.medrag import MedRAG
from src.adapt_medrag import AdaptiveMedRAG


load_dotenv()



NUM_SAMPLES = 1

START_INDEX = 585

N_RUNS = 1

#MODEL_NAME = (
#    "chabokan/gpt-4o-mini"
#)

MODEL_NAME = "ArvanCloud/GPT-4o-mini"

RETRIEVER_NAME = "MedCPT"

CORPUS_NAME = "Textbooks"

K = 10

SEED = 42

STANDARD_ROUNDS = 3

STANDARD_QUERIES_PER_ROUND = 2

ADAPTIVE_MAX_ITERATIONS = 3

ADAPTIVE_INITIAL_QUERIES = 2

CONFIDENCE_THRESHOLD = 85

STOP_RAG_CHECKPOINT = (
    "./stop_rag_ckpt/"
    "best_stop_rag_qnetwork.pt"
)

STOP_RAG_MODEL = (
    "michiyasunaga/BioLinkBERT-base"
)

STOP_RAG_MAX_LENGTH = 512

OUTPUT_FILE = (
    "experiment_results_test_repair_514q.json"
)



def parse_predicted_letter(raw_answer):
    """
    Robustly extract A-F from the final model response.
    """

    if raw_answer is None:
        return "UNKNOWN"

    text = str(raw_answer).strip()



    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    ).strip()


    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            value = str(
                parsed.get(
                    "answer",
                    ""
                )
            ).strip().upper()

            match = re.match(
                r"^([A-F])(?:[\.\)\:\s*]|$)",
                value
            )

            if match:
                return match.group(1)

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):
        pass

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, dict):

            value = str(
                parsed.get(
                    "answer",
                    ""
                )
            ).strip().upper()

            match = re.match(
                r"^([A-F])(?:[\.\)\:\s*]|$)",
                value
            )

            if match:
                return match.group(1)

    except (
        ValueError,
        SyntaxError
    ):
        pass


    match = re.search(
        r"\b(?:answer|correct answer)\s*"
        r"(?:is|:)\s*([A-F])\b",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()


    match = re.search(
        r"(?m)^\s*([A-F])[\.\)\:]\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()


    match = re.fullmatch(
        r"\s*([A-F])\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return "UNKNOWN"



def load_medqa_subset(
    num_samples=50
):
    print(
        f"Loading {num_samples} MedQA questions..."
    )

    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="train"
    )

    dataset = dataset.shuffle(
        seed=SEED
    )

    subset = dataset.select(
        range(START_INDEX, num_samples+START_INDEX)
    )



    return [
        {
            "id": f"MedQA_{START_INDEX+ i + 1}",

            "question":
                item["question"],

            "options":
                item["options"],

            "ground_truth":
                item["answer_idx"]
        }
        for i, item in enumerate(subset)
    ]




def run_standard_benchmark(
    dataset,
    medrag_runner,
    fixed_rounds=3,
    n_queries=2,
    k=10
):

    results = []

    correct = 0

    total_iterations = 0
    total_queries = 0

    total_successful_retrievals = 0
    total_failed_retrievals = 0

    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    total_latency = 0.0

    print("\n")
    print("=" * 100)
    print("STANDARD i-MedRAG")
    print("=" * 100)

    for item in dataset:

        print("\n")
        print("-" * 100)

        print(
            f"Evaluating "
            f"{item['id']} | STANDARD"
        )

        print("-" * 100)

        question_start = time.perf_counter()

        try:

            (
                raw_answer,
                history,
                metrics
            ) = medrag_runner.i_medrag_answer(
                question=item["question"],
                options=item["options"],
                k=k,
                n_rounds=fixed_rounds,
                n_queries=n_queries,
            )

            latency_seconds = (
                time.perf_counter()
                - question_start
            )

            predicted = parse_predicted_letter(
                raw_answer
            )

            is_correct = (
                predicted
                == item["ground_truth"]
            )

            if is_correct:
                correct += 1

            question_queries = (
                fixed_rounds * n_queries
            )

            llm_calls = metrics.get(
                "llm_calls",
                0
            )

            prompt_tokens = metrics.get(
                "prompt_tokens",
                0
            )

            completion_tokens = metrics.get(
                "completion_tokens",
                0
            )

            total_question_tokens = metrics.get(
                "total_tokens",
                0
            )

            total_iterations += fixed_rounds

            total_queries += question_queries

            total_successful_retrievals += (
                question_queries
            )

            total_llm_calls += llm_calls

            total_prompt_tokens += (
                prompt_tokens
            )

            total_completion_tokens += (
                completion_tokens
            )

            total_tokens += (
                total_question_tokens
            )

            total_latency += latency_seconds

            result = {
                "id":
                    item["id"],

                "configuration":
                    "standard",

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                "predicted":
                    predicted,

                "correct":
                    is_correct,

                "raw_answer":
                    raw_answer,

                "iterations":
                    fixed_rounds,

                "queries_per_iteration":
                    n_queries,

                "queries_generated":
                    question_queries,

                "successful_retrievals":
                    question_queries,

                "failed_retrievals":
                    0,

                "llm_calls":
                    llm_calls,

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "total_tokens":
                    total_question_tokens,

                "latency_seconds":
                    latency_seconds,

                "history":
                    history
            }

        except Exception as e:

            latency_seconds = (
                time.perf_counter()
                - question_start
            )

            result = {
                "id":
                    item["id"],

                "configuration":
                    "standard",

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                "predicted":
                    "ERROR",

                "correct":
                    False,

                "raw_answer":
                    None,

                "error":
                    str(e),

                "iterations":
                    0,

                "queries_per_iteration":
                    n_queries,

                "queries_generated":
                    0,

                "successful_retrievals":
                    0,

                "failed_retrievals":
                    fixed_rounds * n_queries,

                "llm_calls":
                    getattr(
                        medrag_runner,
                        "llm_call_count",
                        0
                    ),

                "prompt_tokens":
                    getattr(
                        medrag_runner,
                        "prompt_tokens",
                        0
                    ),

                "completion_tokens":
                    getattr(
                        medrag_runner,
                        "completion_tokens",
                        0
                    ),

                "total_tokens":
                    getattr(
                        medrag_runner,
                        "total_tokens",
                        0
                    ),

                "latency_seconds":
                    latency_seconds,

                "history":
                    []
            }

            total_failed_retrievals += (
                fixed_rounds * n_queries
            )

            total_llm_calls += (
                result["llm_calls"]
            )

            total_prompt_tokens += (
                result["prompt_tokens"]
            )

            total_completion_tokens += (
                result["completion_tokens"]
            )

            total_tokens += (
                result["total_tokens"]
            )

            total_latency += (
                latency_seconds
            )

        results.append(result)

        print(
            f"\nPredicted: "
            f"{result['predicted']}"
        )

        print(
            f"Expected: "
            f"{result['expected']}"
        )

        print(
            f"Correct: "
            f"{result['correct']}"
        )

        print(
            f"Iterations: "
            f"{result['iterations']}"
        )

        print(
            f"Queries: "
            f"{result['queries_generated']}"
        )

        print(
            f"LLM calls: "
            f"{result['llm_calls']}"
        )

        print(
            f"Total tokens: "
            f"{result['total_tokens']}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']:.2f}s"
        )

    n = len(dataset)

    summary = {

        "configuration":
            "standard",

        "accuracy":
            correct / n * 100
            if n > 0
            else 0.0,

        "correct":
            correct,

        "total_questions":
            n,

        "avg_iterations":
            total_iterations / n
            if n > 0
            else 0.0,

        "total_iterations":
            total_iterations,

        "avg_queries":
            total_queries / n
            if n > 0
            else 0.0,

        "total_queries":
            total_queries,

        "avg_successful_retrievals":
            total_successful_retrievals / n
            if n > 0
            else 0.0,

        "total_successful_retrievals":
            total_successful_retrievals,

        "avg_failed_retrievals":
            total_failed_retrievals / n
            if n > 0
            else 0.0,

        "total_failed_retrievals":
            total_failed_retrievals,

        "avg_llm_calls":
            total_llm_calls / n
            if n > 0
            else 0.0,

        "total_llm_calls":
            total_llm_calls,

        "avg_prompt_tokens":
            total_prompt_tokens / n
            if n > 0
            else 0.0,

        "total_prompt_tokens":
            total_prompt_tokens,

        "avg_completion_tokens":
            total_completion_tokens / n
            if n > 0
            else 0.0,

        "total_completion_tokens":
            total_completion_tokens,

        "avg_total_tokens":
            total_tokens / n
            if n > 0
            else 0.0,

        "total_tokens":
            total_tokens,

        "avg_latency_seconds":
            total_latency / n
            if n > 0
            else 0.0,

        "total_latency_seconds":
            total_latency,

        "results":
            results
    }

    print("\n")
    print("=" * 100)
    print("STANDARD SUMMARY")
    print("=" * 100)

    print(
        f"Accuracy: "
        f"{summary['accuracy']:.2f}% "
        f"({correct}/{n})"
    )

    print(
        f"Average iterations: "
        f"{summary['avg_iterations']:.2f}"
    )

    print(
        f"Average queries: "
        f"{summary['avg_queries']:.2f}"
    )

    print(
        f"Average LLM calls: "
        f"{summary['avg_llm_calls']:.2f}"
    )

    print(
        f"Average prompt tokens: "
        f"{summary['avg_prompt_tokens']:.2f}"
    )

    print(
        f"Average completion tokens: "
        f"{summary['avg_completion_tokens']:.2f}"
    )

    print(
        f"Average total tokens: "
        f"{summary['avg_total_tokens']:.2f}"
    )

    print(
        f"Average latency: "
        f"{summary['avg_latency_seconds']:.2f}s"
    )

    print("=" * 100)

    return summary



def run_adaptive_benchmark(
    dataset,
    medrag_runner,
    config_name="confidence",
    use_planner=False,
    use_planner_budget=False,
    use_coverage=False,
    use_novelty=True,
    use_options_in_gate=True,
    gate_features=None,
    query_budget_mode="confidence",
    max_iterations=3,
    initial_queries=2,
    confidence_threshold=85,
    k=10
):

    results = []

    correct = 0

    total_iterations = 0
    total_queries = 0

    total_successful_retrievals = 0
    total_failed_retrievals = 0

    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    total_latency = 0.0

    print("\n")
    print("=" * 100)
    print(
        f"CONFIGURATION: "
        f"{config_name}"
    )
    print("=" * 100)

    print(
        f"Planner: "
        f"{use_planner}"
    )

    print(
        f"Planner Budget: "
        f"{use_planner_budget}"
    )

    print(
        f"Coverage: "
        f"{use_coverage}"
    )

    print(
        f"Novelty: "
        f"{use_novelty}"
    )

    print(
        f"Options In Gate: "
        f"{use_options_in_gate}"
    )

    print(
        f"Gate Features: "
        f"{gate_features}"
    )

    for item in dataset:

        print("\n")
        print("-" * 100)

        print(
            f"Evaluating "
            f"{item['id']} | "
            f"{config_name}"
        )

        print("-" * 100)

        question_start = time.perf_counter()

        try:

            (
                raw_answer,
                history,
                iterations,
                confidence,
                query_budget_history,
                queries_generated,
                successful_retrievals,
                failed_retrievals,
                evidence_history,
                coverage_history,
                controller_config
            ) = medrag_runner.adaptive_answer(

                question=item["question"],

                options=item["options"],

                k=k,

                max_iterations=max_iterations,

                initial_queries=initial_queries,

                confidence_threshold=confidence_threshold,

                use_planner=use_planner,

                use_planner_budget=
                    use_planner_budget,

                use_coverage=
                    use_coverage,

                use_novelty=
                    use_novelty,

                use_options_in_gate=
                    use_options_in_gate,

                gate_features=
                    gate_features,

                query_budget_mode=
                    query_budget_mode,

            )

            latency_seconds = (
                time.perf_counter()
                - question_start
            )

            predicted = parse_predicted_letter(
                raw_answer
            )

            is_correct = (
                predicted
                == item["ground_truth"]
            )

            if is_correct:
                correct += 1

            confidence_history = (
                controller_config.get(
                    "confidence_history",
                    []
                )
            )

            novelty_history = (
                controller_config.get(
                    "novelty_history",
                    []
                )
            )

            discrimination_history = (
                controller_config.get(
                    "discrimination_history",
                    []
                )
            )

            gate_reasoning_history = (
                controller_config.get(
                    "gate_reasoning_history",
                    []
                )
            )

            pre_retrieval_confidence = (
                controller_config.get(
                    "pre_retrieval_confidence",
                    None
                )
            )

            pre_retrieval_answerable = (
                controller_config.get(
                    "pre_retrieval_answerable",
                    None
                )
            )

            pre_retrieval_discrimination = (
                controller_config.get(
                    "pre_retrieval_discrimination",
                    None
                )
            )

            pre_retrieval_stopped = (
                controller_config.get(
                    "pre_retrieval_stopped",
                    False
                )
            )

            llm_calls = (
                controller_config.get(
                    "llm_calls",
                    0
                )
            )

            prompt_tokens = (
                controller_config.get(
                    "prompt_tokens",
                    0
                )
            )

            completion_tokens = (
                controller_config.get(
                    "completion_tokens",
                    0
                )
            )

            total_question_tokens = (
                controller_config.get(
                    "total_tokens",
                    0
                )
            )

            total_iterations += iterations

            total_queries += queries_generated

            total_successful_retrievals += (
                successful_retrievals
            )

            total_failed_retrievals += (
                failed_retrievals
            )

            total_llm_calls += llm_calls

            total_prompt_tokens += prompt_tokens

            total_completion_tokens += (
                completion_tokens
            )

            total_tokens += total_question_tokens

            total_latency += latency_seconds

            result = {

                "id":
                    item["id"],

                "configuration":
                    config_name,

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                "predicted":
                    predicted,

                "correct":
                    is_correct,

                "raw_answer":
                    raw_answer,

                "iterations":
                    iterations,

                "confidence":
                    confidence,

                "confidence_history":
                    confidence_history,

                "novelty_history":
                    novelty_history,

                "discrimination_history":
                    discrimination_history,

                "gate_reasoning_history":
                    gate_reasoning_history,

                "pre_retrieval_confidence":
                    pre_retrieval_confidence,

                "pre_retrieval_answerable":
                    pre_retrieval_answerable,

                "pre_retrieval_discrimination":
                    pre_retrieval_discrimination,

                "pre_retrieval_stopped":
                    pre_retrieval_stopped,

                "query_budget_history":
                    query_budget_history,

                "queries_generated":
                    queries_generated,

                "successful_retrievals":
                    successful_retrievals,

                "failed_retrievals":
                    failed_retrievals,

                "use_planner":
                    use_planner,

                "use_planner_budget":
                    use_planner_budget,

                "use_coverage":
                    use_coverage,

                "use_novelty":
                    use_novelty,

                "use_options_in_gate":
                    use_options_in_gate,

                "gate_features":
                    gate_features,

                "query_budget_mode":
                    query_budget_mode,

                "evidence_history":
                    evidence_history,

                "coverage_history":
                    coverage_history,

                "controller_config":
                    controller_config,

                "llm_calls":
                    llm_calls,

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "total_tokens":
                    total_question_tokens,

                "latency_seconds":
                    latency_seconds,

                "history":
                    history
            }

        except Exception as e:

            latency_seconds = (
                time.perf_counter()
                - question_start
            )

            result = {

                "id":
                    item["id"],

                "configuration":
                    config_name,

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                "predicted":
                    "ERROR",

                "correct":
                    False,

                "raw_answer":
                    None,

                "error":
                    str(e),

                "iterations":
                    0,

                "confidence":
                    None,

                "confidence_history":
                    [],

                "novelty_history":
                    [],

                "discrimination_history":
                    [],

                "gate_reasoning_history":
                    [],

                "pre_retrieval_confidence":
                    None,

                "pre_retrieval_answerable":
                    None,

                "pre_retrieval_discrimination":
                    None,

                "pre_retrieval_stopped":
                    False,

                "query_budget_history":
                    [],

                "queries_generated":
                    0,

                "successful_retrievals":
                    0,

                "failed_retrievals":
                    0,

                "use_planner":
                    use_planner,

                "use_planner_budget":
                    use_planner_budget,

                "use_coverage":
                    use_coverage,

                "use_novelty":
                    use_novelty,

                "use_options_in_gate":
                    use_options_in_gate,

                "gate_features":
                    gate_features,

                "query_budget_mode":
                    query_budget_mode,

                "evidence_history":
                    [],

                "coverage_history":
                    [],

                "controller_config":
                    {},

                "llm_calls":
                    getattr(
                        medrag_runner,
                        "llm_call_count",
                        0
                    ),

                "prompt_tokens":
                    getattr(
                        medrag_runner,
                        "prompt_tokens",
                        0
                    ),

                "completion_tokens":
                    getattr(
                        medrag_runner,
                        "completion_tokens",
                        0
                    ),

                "total_tokens":
                    getattr(
                        medrag_runner,
                        "total_tokens",
                        0
                    ),

                "latency_seconds":
                    latency_seconds,

                "history":
                    []
            }

            total_llm_calls += result["llm_calls"]

            total_prompt_tokens += (
                result["prompt_tokens"]
            )

            total_completion_tokens += (
                result["completion_tokens"]
            )

            total_tokens += result["total_tokens"]

            total_latency += latency_seconds

        results.append(result)

        print(
            f"\nPredicted: "
            f"{result['predicted']}"
        )

        print(
            f"Expected: "
            f"{result['expected']}"
        )

        print(
            f"Correct: "
            f"{result['correct']}"
        )

        print(
            f"Iterations: "
            f"{result['iterations']}"
        )

        print(
            f"Queries generated: "
            f"{result['queries_generated']}"
        )

        print(
            f"Successful retrievals: "
            f"{result['successful_retrievals']}"
        )

        print(
            f"Failed retrievals: "
            f"{result['failed_retrievals']}"
        )

        print(
            f"Budget history: "
            f"{result['query_budget_history']}"
        )

        print(
            f"Confidence history: "
            f"{result['confidence_history']}"
        )

        print(
            f"Discrimination history: "
            f"{result['discrimination_history']}"
        )

        print(
            f"Pre-retrieval confidence: "
            f"{result['pre_retrieval_confidence']}"
        )

        print(
            f"Pre-retrieval stopped: "
            f"{result['pre_retrieval_stopped']}"
        )

        print(
            f"LLM calls: "
            f"{result['llm_calls']}"
        )

        print(
            f"Total tokens: "
            f"{result['total_tokens']}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']:.2f}s"
        )

    n = len(dataset)

    summary = {

        "configuration":
            config_name,

        "accuracy":
            correct / n * 100
            if n > 0
            else 0.0,

        "correct":
            correct,

        "total_questions":
            n,

        "avg_iterations":
            total_iterations / n
            if n > 0
            else 0.0,

        "total_iterations":
            total_iterations,

        "avg_queries":
            total_queries / n
            if n > 0
            else 0.0,

        "total_queries":
            total_queries,

        "avg_successful_retrievals":
            total_successful_retrievals / n
            if n > 0
            else 0.0,

        "total_successful_retrievals":
            total_successful_retrievals,

        "avg_failed_retrievals":
            total_failed_retrievals / n
            if n > 0
            else 0.0,

        "total_failed_retrievals":
            total_failed_retrievals,

        "avg_llm_calls":
            total_llm_calls / n
            if n > 0
            else 0.0,

        "total_llm_calls":
            total_llm_calls,

        "avg_prompt_tokens":
            total_prompt_tokens / n
            if n > 0
            else 0.0,

        "total_prompt_tokens":
            total_prompt_tokens,

        "avg_completion_tokens":
            total_completion_tokens / n
            if n > 0
            else 0.0,

        "total_completion_tokens":
            total_completion_tokens,

        "avg_total_tokens":
            total_tokens / n
            if n > 0
            else 0.0,

        "total_tokens":
            total_tokens,

        "avg_latency_seconds":
            total_latency / n
            if n > 0
            else 0.0,

        "total_latency_seconds":
            total_latency,

        "results":
            results
    }

    print("\n")
    print("=" * 100)

    print(
        f"SUMMARY: "
        f"{config_name}"
    )

    print("=" * 100)

    print(
        f"Accuracy: "
        f"{summary['accuracy']:.2f}% "
        f"({correct}/{n})"
    )

    print(
        f"Average iterations: "
        f"{summary['avg_iterations']:.2f}"
    )

    print(
        f"Average queries: "
        f"{summary['avg_queries']:.2f}"
    )

    print(
        f"Average LLM calls: "
        f"{summary['avg_llm_calls']:.2f}"
    )

    print(
        f"Average prompt tokens: "
        f"{summary['avg_prompt_tokens']:.2f}"
    )

    print(
        f"Average completion tokens: "
        f"{summary['avg_completion_tokens']:.2f}"
    )

    print(
        f"Average total tokens: "
        f"{summary['avg_total_tokens']:.2f}"
    )

    print(
        f"Average latency: "
        f"{summary['avg_latency_seconds']:.2f}s"
    )

    print("=" * 100)

    return summary




def run_trained_stop_rag_benchmark(
    dataset,
    medrag_runner,
    max_iterations=3,
    initial_queries=2,
    k=10
):

    results = []

    correct = 0

    total_iterations = 0
    total_queries = 0

    total_successful_retrievals = 0
    total_failed_retrievals = 0

    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    total_latency = 0.0

    print("\n")
    print("=" * 110)
    print(
        "LEARNED STOP-RAG BENCHMARK"
    )
    print("=" * 110)

    for item in dataset:

        print("\n")
        print("-" * 110)

        print(
            f"Evaluating "
            f"{item['id']} | "
            f"LEARNED STOP-RAG"
        )

        print("-" * 110)

        question_start = time.perf_counter()

        try:

            (
                raw_answer,
                history,
                iterations,
                query_budget_history,
                queries_generated,
                successful_retrievals,
                failed_retrievals,
                q_history,
                controller_config
            ) = medrag_runner.learned_stop_rag_answer(

                question=item["question"],

                options=item["options"],

                k=k,

                max_iterations=max_iterations,

                initial_queries=initial_queries,


            )

            latency_seconds = (
                time.perf_counter()
                - question_start
            )

            predicted = parse_predicted_letter(
                raw_answer
            )

            is_correct = (
                predicted
                == item["ground_truth"]
            )

            if is_correct:
                correct += 1

            llm_calls = (
                controller_config.get(
                    "llm_calls",
                    0
                )
            )

            prompt_tokens = (
                controller_config.get(
                    "prompt_tokens",
                    0
                )
            )

            completion_tokens = (
                controller_config.get(
                    "completion_tokens",
                    0
                )
            )

            total_question_tokens = (
                controller_config.get(
                    "total_tokens",
                    0
                )
            )

            total_iterations += iterations

            total_queries += queries_generated

            total_successful_retrievals += (
                successful_retrievals
            )

            total_failed_retrievals += (
                failed_retrievals
            )

            total_llm_calls += llm_calls

            total_prompt_tokens += prompt_tokens

            total_completion_tokens += (
                completion_tokens
            )

            total_tokens += (
                total_question_tokens
            )

            total_latency += latency_seconds

            result = {

                "id":
                    item["id"],

                "configuration":
                    "trained_stop_rag",

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                "predicted":
                    predicted,

                "correct":
                    is_correct,

                "raw_answer":
                    raw_answer,

                "iterations":
                    iterations,

                "query_budget_history":
                    query_budget_history,

                "queries_generated":
                    queries_generated,

                "successful_retrievals":
                    successful_retrievals,

                "failed_retrievals":
                    failed_retrievals,

                "q_history":
                    q_history,

                "controller_config":
                    controller_config,

                "llm_calls":
                    llm_calls,

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "total_tokens":
                    total_question_tokens,

                "latency_seconds":
                    latency_seconds,

                "history":
                    history
            }

        except Exception as e:

            latency_seconds = (
                time.perf_counter()
                - question_start
            )

            result = {

                "id":
                    item["id"],

                "configuration":
                    "trained_stop_rag",

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                "predicted":
                    "ERROR",

                "correct":
                    False,

                "raw_answer":
                    None,

                "error":
                    str(e),

                "iterations":
                    0,

                "query_budget_history":
                    [],

                "queries_generated":
                    0,

                "successful_retrievals":
                    0,

                "failed_retrievals":
                    0,

                "q_history":
                    [],

                "controller_config":
                    {},

                "llm_calls":
                    getattr(
                        medrag_runner,
                        "llm_call_count",
                        0
                    ),

                "prompt_tokens":
                    getattr(
                        medrag_runner,
                        "prompt_tokens",
                        0
                    ),

                "completion_tokens":
                    getattr(
                        medrag_runner,
                        "completion_tokens",
                        0
                    ),

                "total_tokens":
                    getattr(
                        medrag_runner,
                        "total_tokens",
                        0
                    ),

                "latency_seconds":
                    latency_seconds,

                "history":
                    []
            }

            total_llm_calls += (
                result["llm_calls"]
            )

            total_prompt_tokens += (
                result["prompt_tokens"]
            )

            total_completion_tokens += (
                result["completion_tokens"]
            )

            total_tokens += (
                result["total_tokens"]
            )

            total_latency += (
                latency_seconds
            )

        results.append(result)

        print(
            f"\nPredicted: "
            f"{result['predicted']}"
        )

        print(
            f"Expected: "
            f"{result['expected']}"
        )

        print(
            f"Correct: "
            f"{result['correct']}"
        )

        print(
            f"Iterations: "
            f"{result['iterations']}"
        )

        print(
            f"Queries: "
            f"{result['queries_generated']}"
        )

        print(
            f"Successful retrievals: "
            f"{result['successful_retrievals']}"
        )

        print(
            f"LLM calls: "
            f"{result['llm_calls']}"
        )

        print(
            f"Total tokens: "
            f"{result['total_tokens']}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']:.2f}s"
        )

    n = len(dataset)

    summary = {

        "configuration":
            "trained_stop_rag",

        "accuracy":
            correct / n * 100
            if n > 0
            else 0.0,

        "correct":
            correct,

        "total_questions":
            n,

        "avg_iterations":
            total_iterations / n
            if n > 0
            else 0.0,

        "total_iterations":
            total_iterations,

        "avg_queries":
            total_queries / n
            if n > 0
            else 0.0,

        "total_queries":
            total_queries,

        "avg_successful_retrievals":
            total_successful_retrievals / n
            if n > 0
            else 0.0,

        "total_successful_retrievals":
            total_successful_retrievals,

        "avg_failed_retrievals":
            total_failed_retrievals / n
            if n > 0
            else 0.0,

        "total_failed_retrievals":
            total_failed_retrievals,

        "avg_llm_calls":
            total_llm_calls / n
            if n > 0
            else 0.0,

        "total_llm_calls":
            total_llm_calls,

        "avg_prompt_tokens":
            total_prompt_tokens / n
            if n > 0
            else 0.0,

        "total_prompt_tokens":
            total_prompt_tokens,

        "avg_completion_tokens":
            total_completion_tokens / n
            if n > 0
            else 0.0,

        "total_completion_tokens":
            total_completion_tokens,

        "avg_total_tokens":
            total_tokens / n
            if n > 0
            else 0.0,

        "total_tokens":
            total_tokens,

        "avg_latency_seconds":
            total_latency / n
            if n > 0
            else 0.0,

        "total_latency_seconds":
            total_latency,

        "results":
            results
    }

    print("\n")
    print("=" * 110)
    print(
        "LEARNED STOP-RAG SUMMARY"
    )
    print("=" * 110)

    print(
        f"Accuracy: "
        f"{summary['accuracy']:.2f}% "
        f"({correct}/{n})"
    )

    print(
        f"Average iterations: "
        f"{summary['avg_iterations']:.2f}"
    )

    print(
        f"Average queries: "
        f"{summary['avg_queries']:.2f}"
    )

    print(
        f"Average LLM calls: "
        f"{summary['avg_llm_calls']:.2f}"
    )

    print(
        f"Average total tokens: "
        f"{summary['avg_total_tokens']:.2f}"
    )

    print(
        f"Average latency: "
        f"{summary['avg_latency_seconds']:.2f}s"
    )

    print("=" * 110)

    return summary




def print_three_way_comparison(
    standard_result,
    confidence_result,
    trained_result
):

    print("\n")
    print("=" * 115)
    print("THREE-WAY COMPARISON")
    print("=" * 115)

    print(
        f"{'Metric':<35}"
        f"{'Standard':<20}"
        f"{'Confidence':<20}"
        f"{'Learned Stop-RAG':<25}"
    )

    print("-" * 115)

    metrics = [

        (
            "Accuracy (%)",
            "accuracy"
        ),

        (
            "Average iterations",
            "avg_iterations"
        ),

        (
            "Average queries",
            "avg_queries"
        ),

        (
            "Average LLM calls",
            "avg_llm_calls"
        ),

        (
            "Average prompt tokens",
            "avg_prompt_tokens"
        ),

        (
            "Average completion tokens",
            "avg_completion_tokens"
        ),

        (
            "Average total tokens",
            "avg_total_tokens"
        ),

        (
            "Average latency (s)",
            "avg_latency_seconds"
        )
    ]

    for label, key in metrics:

        print(
            f"{label:<35}"
            f"{standard_result[key]:<20.2f}"
            f"{confidence_result[key]:<20.2f}"
            f"{trained_result[key]:<25.2f}"
        )

    print("=" * 115)




def safe_paired_test(
    baseline_values,
    proposed_values
):

    baseline = np.asarray(
        baseline_values,
        dtype=float
    )

    proposed = np.asarray(
        proposed_values,
        dtype=float
    )

    if len(baseline) != len(proposed):
        raise ValueError(
            "Paired arrays must have equal length."
        )

    if len(baseline) < 2:
        return {
            "t_statistic": None,
            "p_value_ttest": None,
            "wilcoxon_statistic": None,
            "p_value_wilcoxon": None
        }


    t_result = ttest_rel(
        proposed,
        baseline
    )

    t_statistic = (
        None
        if np.isnan(t_result.statistic)
        else float(t_result.statistic)
    )

    p_value_ttest = (
        None
        if np.isnan(t_result.pvalue)
        else float(t_result.pvalue)
    )

    

    try:

        w_result = wilcoxon(
            proposed,
            baseline,
            zero_method="wilcox",
            alternative="two-sided"
        )

        w_statistic = float(
            w_result.statistic
        )

        p_value_wilcoxon = float(
            w_result.pvalue
        )

    except ValueError:

        w_statistic = None
        p_value_wilcoxon = None

    return {

        "t_statistic":
            t_statistic,

        "p_value_ttest":
            p_value_ttest,

        "wilcoxon_statistic":
            w_statistic,

        "p_value_wilcoxon":
            p_value_wilcoxon
    }




def mcnemar_accuracy_test(
    baseline_result,
    proposed_result
):
    """
    Accuracy comparison for paired questions
    within one run.

    b01:
        baseline correct, proposed wrong

    b10:
        baseline wrong, proposed correct
    """

    baseline = {
        x["id"]: x
        for x in baseline_result["results"]
    }

    proposed = {
        x["id"]: x
        for x in proposed_result["results"]
    }

    common_ids = (
        set(baseline)
        & set(proposed)
    )

    baseline_correct_proposed_wrong = 0
    baseline_wrong_proposed_correct = 0

    for qid in common_ids:

        b = bool(
            baseline[qid]["correct"]
        )

        p = bool(
            proposed[qid]["correct"]
        )

        if b and not p:

            baseline_correct_proposed_wrong += 1

        elif not b and p:

            baseline_wrong_proposed_correct += 1

    discordant = (
        baseline_correct_proposed_wrong
        +
        baseline_wrong_proposed_correct
    )

    if discordant == 0:

        p_value = 1.0

    else:

        # Exact two-sided McNemar test
        test = binomtest(
            baseline_wrong_proposed_correct,
            discordant,
            p=0.5,
            alternative="two-sided"
        )

        p_value = float(
            test.pvalue
        )

    return {

        "baseline_correct_proposed_wrong":
            baseline_correct_proposed_wrong,

        "baseline_wrong_proposed_correct":
            baseline_wrong_proposed_correct,

        "discordant_pairs":
            discordant,

        "p_value":
            p_value
    }




def analyze_five_runs(
    all_runs,
    baseline_name,
    proposed_name
):
    """
    all_runs:

    [
        {
            "standard": ...,
            "confidence_adaptive": ...,
            "trained_stop_rag": ...
        },
        ...
    ]
    """

    metric_definitions = [

        (
            "accuracy",
            "Accuracy"
        ),

        (
            "avg_iterations",
            "Iterations"
        ),

        (
            "avg_queries",
            "Queries"
        ),

        (
            "avg_llm_calls",
            "LLM calls"
        ),

        (
            "avg_prompt_tokens",
            "Prompt tokens"
        ),

        (
            "avg_completion_tokens",
            "Completion tokens"
        ),

        (
            "avg_total_tokens",
            "Total tokens"
        ),

        (
            "avg_latency_seconds",
            "Latency"
        )
    ]

    statistics = {

        "baseline":
            baseline_name,

        "proposed":
            proposed_name,

        "n_runs":
            len(all_runs),

        "metrics":
            {},

        "mcnemar_by_run":
            []
    }


    for metric_key, metric_name in metric_definitions:

        baseline_values = [
            run[baseline_name][metric_key]
            for run in all_runs
        ]

        proposed_values = [
            run[proposed_name][metric_key]
            for run in all_runs
        ]

        test_result = safe_paired_test(
            baseline_values,
            proposed_values
        )

        baseline_array = np.asarray(
            baseline_values,
            dtype=float
        )

        proposed_array = np.asarray(
            proposed_values,
            dtype=float
        )

        test_result.update({

            "metric":
                metric_name,

            "metric_key":
                metric_key,

            "baseline_values":
                [
                    float(x)
                    for x in baseline_values
                ],

            "proposed_values":
                [
                    float(x)
                    for x in proposed_values
                ],

            "baseline_mean":
                float(
                    np.mean(
                        baseline_array
                    )
                ),

            "baseline_std":
                float(
                    np.std(
                        baseline_array,
                        ddof=1
                    )
                ),

            "proposed_mean":
                float(
                    np.mean(
                        proposed_array
                    )
                ),

            "proposed_std":
                float(
                    np.std(
                        proposed_array,
                        ddof=1
                    )
                ),

            "mean_difference":
                float(
                    np.mean(
                        proposed_array
                        -
                        baseline_array
                    )
                )
        })

        statistics[
            "metrics"
        ][metric_key] = test_result


    for run_index, run in enumerate(
        all_runs,
        start=1
    ):

        mcnemar_result = (
            mcnemar_accuracy_test(
                run[baseline_name],
                run[proposed_name]
            )
        )

        mcnemar_result[
            "run"
        ] = run_index

        statistics[
            "mcnemar_by_run"
        ].append(
            mcnemar_result
        )

    

    resource_metrics = [

        "avg_queries",
        "avg_llm_calls",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "avg_latency_seconds"
    ]

    reductions = {}

    for metric in resource_metrics:

        baseline_mean = (
            statistics[
                "metrics"
            ][metric][
                "baseline_mean"
            ]
        )

        proposed_mean = (
            statistics[
                "metrics"
            ][metric][
                "proposed_mean"
            ]
        )

        if baseline_mean != 0:

            reduction = (
                (
                    baseline_mean
                    -
                    proposed_mean
                )
                /
                baseline_mean
                *
                100
            )

        else:

            reduction = 0.0

        reductions[metric] = float(
            reduction
        )

    statistics[
        "relative_reductions_percent"
    ] = reductions

    return statistics




def print_statistical_report(
    statistics
):

    print("\n")
    print("=" * 125)

    print(
        "STATISTICAL SIGNIFICANCE"
    )

    print("=" * 125)

    print(
        f"Baseline: "
        f"{statistics['baseline']}"
    )

    print(
        f"Proposed: "
        f"{statistics['proposed']}"
    )

    print(
        f"Number of runs: "
        f"{statistics['n_runs']}"
    )

    print("\n")

    print(
        f"{'Metric':<25}"
        f"{'Baseline Mean±SD':<25}"
        f"{'Proposed Mean±SD':<25}"
        f"{'Paired t-test p':<20}"
        f"{'Wilcoxon p':<15}"
    )

    print("-" * 125)

    for metric_key, result in (
        statistics["metrics"].items()
    ):

        baseline_text = (
            f"{result['baseline_mean']:.3f} ± "
            f"{result['baseline_std']:.3f}"
        )

        proposed_text = (
            f"{result['proposed_mean']:.3f} ± "
            f"{result['proposed_std']:.3f}"
        )

        p_t = result[
            "p_value_ttest"
        ]

        p_w = result[
            "p_value_wilcoxon"
        ]

        p_t_text = (
            "N/A"
            if p_t is None
            else f"{p_t:.5f}"
        )

        p_w_text = (
            "N/A"
            if p_w is None
            else f"{p_w:.5f}"
        )

        print(
            f"{result['metric']:<25}"
            f"{baseline_text:<25}"
            f"{proposed_text:<25}"
            f"{p_t_text:<20}"
            f"{p_w_text:<15}"
        )

    print("-" * 125)

    print(
        "\nMcNemar accuracy test "
        "(performed separately for each run):"
    )

    for result in statistics[
        "mcnemar_by_run"
    ]:

        print(
            f"  Run {result['run']}: "
            f"baseline-correct/proposed-wrong = "
            f"{result['baseline_correct_proposed_wrong']}, "
            f"baseline-wrong/proposed-correct = "
            f"{result['baseline_wrong_proposed_correct']}, "
            f"p = {result['p_value']:.5f}"
        )

    print(
        "\nRelative resource reductions "
        "(positive = proposed uses less):"
    )

    for metric, reduction in (
        statistics[
            "relative_reductions_percent"
        ].items()
    ):

        print(
            f"  {metric}: "
            f"{reduction:.2f}%"
        )

    print("=" * 125)



def run_one_experiment(
    dataset,
    standard_medrag,
    adaptive_medrag
):

    print("\n")
    print("=" * 120)
    print("RUNNING STANDARD i-MedRAG")
    print("=" * 120)

    standard_result = (
        run_standard_benchmark(

            dataset=dataset,

            medrag_runner=standard_medrag,

            fixed_rounds=
                STANDARD_ROUNDS,

            n_queries=
                STANDARD_QUERIES_PER_ROUND,

            k=K
        )
    )

    print("\n")
    print("=" * 120)
    print(
        "RUNNING CONFIDENCE-ADAPTIVE i-MedRAG"
    )
    print("=" * 120)

    confidence_result = (
        run_adaptive_benchmark(

            dataset=dataset,

            medrag_runner=adaptive_medrag,

            config_name="confidence",

            use_planner=False,

            use_planner_budget=False,

            use_coverage=False,

            use_novelty=True,

            use_options_in_gate=True,

            gate_features=[
                "confidence",
                "answerable",
                "discrimination"
            ],

            query_budget_mode=
                "confidence",

            max_iterations=
                ADAPTIVE_MAX_ITERATIONS,

            initial_queries=
                ADAPTIVE_INITIAL_QUERIES,

            confidence_threshold=
                CONFIDENCE_THRESHOLD,

            k=K
        )
    )

    print("\n")
    print("=" * 120)
    print("RUNNING LEARNED STOP-RAG")
    print("=" * 120)

    trained_result = (
        run_trained_stop_rag_benchmark(

            dataset=dataset,

            medrag_runner=adaptive_medrag,

            max_iterations=
                ADAPTIVE_MAX_ITERATIONS,

            initial_queries=
                ADAPTIVE_INITIAL_QUERIES,

            k=K
        )
    )

    print_three_way_comparison(
        standard_result,
        confidence_result,
        trained_result
    )

    return {

        "standard":
            standard_result,

        "confidence_adaptive":
            confidence_result,

        "trained_stop_rag":
            trained_result
    }



def save_json(
    data,
    filepath
):

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )



if __name__ == "__main__":


    all_questions = load_medqa_subset(
        num_samples=NUM_SAMPLES
    )

    medqa_batch = all_questions[:]

    print("\n")
    print("=" * 100)
    print("QUESTIONS USED")
    print("=" * 100)

    for item in medqa_batch:

        print(
            f"{item['id']} -> "
            f"Ground truth: "
            f"{item['ground_truth']}"
        )


    print("\n")
    print(
        "Initializing STANDARD MedRAG..."
    )

    standard_medrag = MedRAG(

        llm_name=MODEL_NAME,

        rag=True,

        follow_up=True,

        retriever_name=
            RETRIEVER_NAME,

        corpus_name=
            CORPUS_NAME,

        corpus_cache=True
    )


    print("\n")
    print(
        "Initializing ADAPTIVE MedRAG..."
    )

    adaptive_medrag = AdaptiveMedRAG(

        llm_name=MODEL_NAME,

        rag=True,

        follow_up=True,

        retriever_name=
            RETRIEVER_NAME,

        corpus_name=
            CORPUS_NAME,

        corpus_cache=True
    )

    adaptive_medrag.set_stop_rag_controller(

        checkpoint_path=
            STOP_RAG_CHECKPOINT,

        model_name=
            STOP_RAG_MODEL,

        max_length=
            STOP_RAG_MAX_LENGTH
    )

    print(
        "\n[Stop-RAG Controller] Loaded."
    )


    all_runs = []

    for run_id in range(
        1,
        N_RUNS + 1
    ):

        print("\n")
        print("=" * 120)
        print(
            f"EXPERIMENT RUN "
            f"{run_id}/{N_RUNS}"
        )
        print("=" * 120)

        print(
            f"Dataset seed: {SEED}"
        )

        print(
            f"Questions in this run: "
            f"{len(medqa_batch)}"
        )

        try:

            run_result = (
                run_one_experiment(

                    dataset=medqa_batch,

                    standard_medrag=
                        standard_medrag,

                    adaptive_medrag=
                        adaptive_medrag
                )
            )

            run_result[
                "run_id"
            ] = run_id

            run_result[
                "dataset_seed"
            ] = SEED

            all_runs.append(
                run_result
            )


            checkpoint_data = {

                "experiment_settings": {

                    "num_samples":
                        NUM_SAMPLES,

                    "n_runs":
                        N_RUNS,

                    "model":
                        MODEL_NAME,

                    "retriever":
                        RETRIEVER_NAME,

                    "corpus":
                        CORPUS_NAME,

                    "k":
                        K,

                    "seed":
                        SEED,

                    "standard_rounds":
                        STANDARD_ROUNDS,

                    "standard_queries_per_round":
                        STANDARD_QUERIES_PER_ROUND,

                    "adaptive_max_iterations":
                        ADAPTIVE_MAX_ITERATIONS,

                    "adaptive_initial_queries":
                        ADAPTIVE_INITIAL_QUERIES,

                    "confidence_threshold":
                        CONFIDENCE_THRESHOLD,

                    "stop_rag_checkpoint":
                        STOP_RAG_CHECKPOINT,

                    "stop_rag_model":
                        STOP_RAG_MODEL,

                    "stop_rag_max_length":
                        STOP_RAG_MAX_LENGTH
                },

                "questions":
                    medqa_batch,

                "completed_runs":
                    len(all_runs),

                "runs":
                    all_runs
            }

            save_json(
                checkpoint_data,
                OUTPUT_FILE
            )

            print("\n")
            print(
                f"[Checkpoint saved] "
                f"{OUTPUT_FILE}"
            )

        except Exception as e:

            print("\n")
            print("=" * 100)

            print(
                f"RUN {run_id} FAILED"
            )

            print("=" * 100)

            print(
                f"Error: {e}"
            )

            print(
                "Continuing to next run..."
            )


    if len(all_runs) < 2:

        print("\n")
        print(
            "Not enough completed runs "
            "for statistical testing."
        )

        raise SystemExit


    standard_vs_trained = (
        analyze_five_runs(

            all_runs,

            baseline_name=
                "standard",

            proposed_name=
                "trained_stop_rag"
        )
    )


    standard_vs_confidence = (
        analyze_five_runs(

            all_runs,

            baseline_name=
                "standard",

            proposed_name=
                "confidence_adaptive"
        )
    )

    confidence_vs_trained = (
        analyze_five_runs(

            all_runs,

            baseline_name=
                "confidence_adaptive",

            proposed_name=
                "trained_stop_rag"
        )
    )


    print("\n")
    print("=" * 130)

    print(
        "FINAL STATISTICAL ANALYSIS"
    )

    print("=" * 130)

    print_statistical_report(
        standard_vs_trained
    )

    print_statistical_report(
        standard_vs_confidence
    )

    print_statistical_report(
        confidence_vs_trained
    )


    final_output = {

        "experiment_settings": {

            "num_samples":
                NUM_SAMPLES,

            "n_runs":
                N_RUNS,

            "model":
                MODEL_NAME,

            "retriever":
                RETRIEVER_NAME,

            "corpus":
                CORPUS_NAME,

            "k":
                K,

            "seed":
                SEED,

            "standard_rounds":
                STANDARD_ROUNDS,

            "standard_queries_per_round":
                STANDARD_QUERIES_PER_ROUND,

            "adaptive_max_iterations":
                ADAPTIVE_MAX_ITERATIONS,

            "adaptive_initial_queries":
                ADAPTIVE_INITIAL_QUERIES,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "stop_rag_checkpoint":
                STOP_RAG_CHECKPOINT,

            "stop_rag_model":
                STOP_RAG_MODEL,

            "stop_rag_max_length":
                STOP_RAG_MAX_LENGTH
        },

        "questions":
            medqa_batch,

        "completed_runs":
            len(all_runs),

        "runs":
            all_runs,

        "statistics": {

            "standard_vs_trained":
                standard_vs_trained,

            "standard_vs_confidence":
                standard_vs_confidence,

            "confidence_vs_trained":
                confidence_vs_trained
        }
    }

    save_json(
        final_output,
        OUTPUT_FILE
    )

    print("\n")
    print("=" * 130)

    print(
        "FINAL RESULTS SAVED"
    )

    print("=" * 130)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Completed runs: "
        f"{len(all_runs)}/{N_RUNS}"
    )

    print("=" * 130)