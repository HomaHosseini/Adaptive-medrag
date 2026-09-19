
import time
import json
import re
import ast
import os

from datasets import load_dataset
from dotenv import load_dotenv

from src.medrag import MedRAG
from src.adapt_medrag import AdaptiveMedRAG

# os.environ["ARVAN_API_KEY"] = "d60b8d53-5f6c-5eb6-828a-031d503221d7"
# os.environ["ARVAN_ENDPOINT"] = "https://arvancloudai.ir/gateway/models/Llama-3.3-70B-Instruct-R1AI/wXwicG2w6RhESLlIg7KlsetpzCShAX1k0bXHGZLX9nDHLxr4mJwXDMW-2Vunju0LPAXGRBNA8Hg4NhbIgiU-6UwAscgx9QDw1vINQTl1636Buf89X2lEaj3jhmNVJ9Q5qjRgH_G-9CxWta0kJqLV_LtR3XQypA0QnGRFDXtMfKtcE4Q0w5zta-2-Ul-b1rMjyoCaLU2oe4h7NM8-23TJZsUjYp9JrcetdbTBdzh2TDjNK1Iy0I10_c_8RQIN0B9LAEWZihx5ela0-k8AEHGCTwvU5zqzRXjNpwAdEQ/v1"
# #os.environ["ARVAN_ENDPOINT"] = "https://api.arvancloudai.ir/v1/"


load_dotenv()

api_key = os.getenv("ARVAN_API_KEY")
endpoint = os.getenv("ARVAN_ENDPOINT")

# ============================================================
# 1. PARSE MODEL ANSWER
# ============================================================

def parse_predicted_letter(raw_answer):

    if raw_answer is None:
        return "UNKNOWN"

    text = str(raw_answer).strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    ).strip()

    # --------------------------------------------------------
    # 1. JSON
    # --------------------------------------------------------

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
                r"^([A-F])(?:[\.\)\:\s]|$)",
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

    # --------------------------------------------------------
    # 2. Python dictionary
    # --------------------------------------------------------

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
                r"^([A-F])(?:[\.\)\:\s]|$)",
                value
            )

            if match:
                return match.group(1)

    except (
        ValueError,
        SyntaxError
    ):
        pass

    # --------------------------------------------------------
    # 3. Natural language
    # --------------------------------------------------------

    match = re.search(
        r"\b(?:answer|correct answer)\s*(?:is|:)\s*([A-F])\b",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # --------------------------------------------------------
    # 4. A. Option
    # --------------------------------------------------------

    match = re.search(
        r"(?m)^\s*([A-F])[\.\)\:]\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # --------------------------------------------------------
    # 5. Just A
    # --------------------------------------------------------

    match = re.fullmatch(
        r"\s*([A-F])\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return "UNKNOWN"


# ============================================================
# 2. LOAD MEDQA
# ============================================================

def load_medqa_subset(num_samples=50):

    print(
        f"Loading {num_samples} MedQA questions..."
    )

    dataset = load_dataset(
        "TsinghuaC3I/MedXpertQA","Text",
        split="test"
    )

    subset = dataset.select(
        range(num_samples)
    )

    return [
        {
            "id":
                f"MedQA_{i + 1}",

            "question":
                item["question"],

            "options":
                item["options"],

            "ground_truth":
                item["label"]
        }
        for i, item in enumerate(subset)
    ]


# ============================================================
# 3. STANDARD i-MedRAG BENCHMARK
# ============================================================

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
            f"Evaluating {item['id']} | STANDARD"
        )

        print("-" * 100)

        question_start = time.perf_counter()

        try:

            # ------------------------------------------------
            # Current standard return:
            #
            # answer, messages, metrics
            # ------------------------------------------------

            raw_answer, history, metrics = (
                medrag_runner.i_medrag_answer(

                    question=item["question"],

                    options=item["options"],

                    k=k,

                    n_rounds=fixed_rounds,

                    n_queries=n_queries
                )
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
                fixed_rounds
                *
                n_queries
            )

            latency_seconds = metrics.get(
                "latency_seconds",
                time.perf_counter()
                - question_start
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

            total_iterations += (
                fixed_rounds
            )

            total_queries += (
                question_queries
            )

            total_successful_retrievals += (
                question_queries
            )

            total_llm_calls += (
                llm_calls
            )

            total_prompt_tokens += (
                prompt_tokens
            )

            total_completion_tokens += (
                completion_tokens
            )

            total_tokens += (
                total_question_tokens
            )

            total_latency += (
                latency_seconds
            )

            result = {

                "id":
                    item["id"],

                "configuration":
                    "standard",

                # -----------------------------
                # Question
                # -----------------------------

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                # -----------------------------
                # Answer
                # -----------------------------

                "predicted":
                    predicted,

                "correct":
                    is_correct,

                "raw_answer":
                    raw_answer,

                # -----------------------------
                # Retrieval
                # -----------------------------

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

                # -----------------------------
                # LLM metrics
                # -----------------------------

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

                # -----------------------------
                # Full model history
                # -----------------------------

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

        results.append(
            result
        )

        # ====================================================
        # PRINT QUESTION
        # ====================================================

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
            f"Prompt tokens: "
            f"{result['prompt_tokens']}"
        )

        print(
            f"Completion tokens: "
            f"{result['completion_tokens']}"
        )

        print(
            f"Total tokens: "
            f"{result['total_tokens']}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']:.2f}s"
        )

    # ========================================================
    # AGGREGATE
    # ========================================================

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

    # ========================================================
    # SUMMARY
    # ========================================================

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


# ============================================================
# 4. ADAPTIVE BENCHMARK
# ============================================================

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

    # ========================================================
    # EACH QUESTION
    # ========================================================

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
                    query_budget_mode
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

            # ------------------------------------------------
            # Controller histories
            # ------------------------------------------------

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

            # ------------------------------------------------
            # LLM metrics
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Aggregate
            # ------------------------------------------------

            total_iterations += (
                iterations
            )

            total_queries += (
                queries_generated
            )

            total_successful_retrievals += (
                successful_retrievals
            )

            total_failed_retrievals += (
                failed_retrievals
            )

            total_llm_calls += (
                llm_calls
            )

            total_prompt_tokens += (
                prompt_tokens
            )

            total_completion_tokens += (
                completion_tokens
            )

            total_tokens += (
                total_question_tokens
            )

            total_latency += (
                latency_seconds
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            result = {

                "id":
                    item["id"],

                "configuration":
                    config_name,

                # -----------------------------
                # Question
                # -----------------------------

                "question":
                    item["question"],

                "options":
                    item["options"],

                "expected":
                    item["ground_truth"],

                # -----------------------------
                # Answer
                # -----------------------------

                "predicted":
                    predicted,

                "correct":
                    is_correct,

                "raw_answer":
                    raw_answer,

                # -----------------------------
                # Adaptive behavior
                # -----------------------------

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

                "query_budget_history":
                    query_budget_history,

                "queries_generated":
                    queries_generated,

                "successful_retrievals":
                    successful_retrievals,

                "failed_retrievals":
                    failed_retrievals,

                # -----------------------------
                # Configuration
                # -----------------------------

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

                # -----------------------------
                # Evidence
                # -----------------------------

                "evidence_history":
                    evidence_history,

                "coverage_history":
                    coverage_history,

                # -----------------------------
                # Controller
                # -----------------------------

                "controller_config":
                    controller_config,

                # -----------------------------
                # LLM metrics
                # -----------------------------

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

                # -----------------------------
                # Full history
                # -----------------------------

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
                    0.0,

                "confidence_history":
                    [],

                "novelty_history":
                    [],

                "discrimination_history":
                    [],

                "gate_reasoning_history":
                    [],

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

        results.append(
            result
        )

        # ====================================================
        # PRINT
        # ====================================================

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

    # ========================================================
    # AGGREGATE
    # ========================================================

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

    # ========================================================
    # SUMMARY
    # ========================================================

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


# ============================================================
# 5. QUESTION-LEVEL COMPARISON
# ============================================================

def build_question_comparison(
    standard_result,
    adaptive_result
):

    standard_by_id = {
        item["id"]: item
        for item in
        standard_result["results"]
    }

    adaptive_by_id = {
        item["id"]: item
        for item in
        adaptive_result["results"]
    }

    comparison = []

    common_ids = sorted(
        set(
            standard_by_id.keys()
        )
        &
        set(
            adaptive_by_id.keys()
        )
    )

    for qid in common_ids:

        standard = (
            standard_by_id[qid]
        )

        adaptive = (
            adaptive_by_id[qid]
        )

        comparison.append({

            # --------------------------
            # Identity
            # --------------------------

            "id":
                qid,

            "ground_truth":
                standard["expected"],

            # --------------------------
            # Answers
            # --------------------------

            "standard_prediction":
                standard["predicted"],

            "standard_correct":
                standard["correct"],

            "adaptive_prediction":
                adaptive["predicted"],

            "adaptive_correct":
                adaptive["correct"],

            # --------------------------
            # Outcome
            # --------------------------

            "both_correct":
                (
                    standard["correct"]
                    and
                    adaptive["correct"]
                ),

            "both_wrong":
                (
                    not standard["correct"]
                    and
                    not adaptive["correct"]
                ),

            "adaptive_improved":
                (
                    not standard["correct"]
                    and
                    adaptive["correct"]
                ),

            "adaptive_hurt":
                (
                    standard["correct"]
                    and
                    not adaptive["correct"]
                ),

            "same_prediction":
                (
                    standard["predicted"]
                    ==
                    adaptive["predicted"]
                ),

            # --------------------------
            # Standard efficiency
            # --------------------------

            "standard_iterations":
                standard["iterations"],

            "standard_queries":
                standard["queries_generated"],

            "standard_llm_calls":
                standard["llm_calls"],

            "standard_prompt_tokens":
                standard["prompt_tokens"],

            "standard_completion_tokens":
                standard["completion_tokens"],

            "standard_total_tokens":
                standard["total_tokens"],

            "standard_latency_seconds":
                standard["latency_seconds"],

            # --------------------------
            # Adaptive efficiency
            # --------------------------

            "adaptive_iterations":
                adaptive["iterations"],

            "adaptive_queries":
                adaptive["queries_generated"],

            "adaptive_llm_calls":
                adaptive["llm_calls"],

            "adaptive_prompt_tokens":
                adaptive["prompt_tokens"],

            "adaptive_completion_tokens":
                adaptive["completion_tokens"],

            "adaptive_total_tokens":
                adaptive["total_tokens"],

            "adaptive_latency_seconds":
                adaptive["latency_seconds"],

            # --------------------------
            # Controller behavior
            # --------------------------

            "confidence_history":
                adaptive.get(
                    "confidence_history",
                    []
                ),

            "novelty_history":
                adaptive.get(
                    "novelty_history",
                    []
                ),

            "discrimination_history":
                adaptive.get(
                    "discrimination_history",
                    []
                ),

            "query_budget_history":
                adaptive.get(
                    "query_budget_history",
                    []
                ),

            "gate_reasoning_history":
                adaptive.get(
                    "gate_reasoning_history",
                    []
                )
        })

    return comparison


# ============================================================
# 6. AGGREGATE COMPARISON
# ============================================================

def build_summary_comparison(
    standard_result,
    adaptive_result
):

    metrics = [

        "accuracy",
        "avg_iterations",
        "avg_queries",
        "avg_llm_calls",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "avg_latency_seconds"
    ]

    comparison = {}

    for metric in metrics:

        standard_value = (
            standard_result[metric]
        )

        adaptive_value = (
            adaptive_result[metric]
        )

        comparison[metric] = {

            "standard":
                standard_value,

            "adaptive":
                adaptive_value,

            "adaptive_minus_standard":
                adaptive_value - standard_value
        }

    # ========================================================
    # Reduction metrics
    # ========================================================

    standard_queries = (
        standard_result["avg_queries"]
    )

    adaptive_queries = (
        adaptive_result["avg_queries"]
    )

    if standard_queries > 0:

        query_reduction = (
            (
                standard_queries
                -
                adaptive_queries
            )
            /
            standard_queries
            *
            100
        )

    else:

        query_reduction = 0.0

    standard_tokens = (
        standard_result["avg_total_tokens"]
    )

    adaptive_tokens = (
        adaptive_result["avg_total_tokens"]
    )

    if standard_tokens > 0:

        token_reduction = (
            (
                standard_tokens
                -
                adaptive_tokens
            )
            /
            standard_tokens
            *
            100
        )

    else:

        token_reduction = 0.0

    standard_latency = (
        standard_result[
            "avg_latency_seconds"
        ]
    )

    adaptive_latency = (
        adaptive_result[
            "avg_latency_seconds"
        ]
    )

    if standard_latency > 0:

        latency_reduction = (
            (
                standard_latency
                -
                adaptive_latency
            )
            /
            standard_latency
            *
            100
        )

    else:

        latency_reduction = 0.0

    comparison[
        "query_reduction_percent"
    ] = query_reduction

    comparison[
        "token_reduction_percent"
    ] = token_reduction

    comparison[
        "latency_reduction_percent"
    ] = latency_reduction

    return comparison


# ============================================================
# 7. PRINT COMPARISON
# ============================================================

def print_summary_comparison(
    standard_result,
    adaptive_result
):

    comparison = build_summary_comparison(
        standard_result,
        adaptive_result
    )

    print("\n")
    print("=" * 115)
    print(
        "STANDARD vs CONFIDENCE-ADAPTIVE"
    )
    print("=" * 115)

    print(
        f"{'Metric':<35}"
        f"{'Standard':<20}"
        f"{'Adaptive':<20}"
        f"{'Adaptive - Standard':<25}"
    )

    print("-" * 115)

    rows = [

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

    for label, key in rows:

        print(

            f"{label:<35}"

            f"{comparison[key]['standard']:<20.2f}"

            f"{comparison[key]['adaptive']:<20.2f}"

            f"{comparison[key]['adaptive_minus_standard']:<25.2f}"
        )

    print("-" * 115)

    print(
        f"{'Query reduction (%)':<35}"
        f"{'':<20}"
        f"{'':<20}"
        f"{comparison['query_reduction_percent']:<25.2f}"
    )

    print(
        f"{'Token reduction (%)':<35}"
        f"{'':<20}"
        f"{'':<20}"
        f"{comparison['token_reduction_percent']:<25.2f}"
    )

    print(
        f"{'Latency reduction (%)':<35}"
        f"{'':<20}"
        f"{'':<20}"
        f"{comparison['latency_reduction_percent']:<25.2f}"
    )

    print("=" * 115)


# ============================================================
# 8. PRINT QUESTION-LEVEL OUTCOMES
# ============================================================

def print_question_outcomes(
    comparison
):

    print("\n")
    print("=" * 110)
    print(
        "QUESTION-LEVEL OUTCOMES"
    )
    print("=" * 110)

    print(
        f"{'ID':<12}"
        f"{'GT':<6}"
        f"{'Standard':<12}"
        f"{'Adaptive':<12}"
        f"{'Outcome':<20}"
        f"{'Std Q':<8}"
        f"{'Adp Q':<8}"
    )

    print("-" * 110)

    for item in comparison:

        if item["adaptive_improved"]:

            outcome = (
                "ADAPTIVE IMPROVED"
            )

        elif item["adaptive_hurt"]:

            outcome = (
                "ADAPTIVE HURT"
            )

        elif item["both_correct"]:

            outcome = (
                "BOTH CORRECT"
            )

        elif item["both_wrong"]:

            outcome = (
                "BOTH WRONG"
            )

        else:

            outcome = "OTHER"

        print(

            f"{item['id']:<12}"

            f"{item['ground_truth']:<6}"

            f"{item['standard_prediction']:<12}"

            f"{item['adaptive_prediction']:<12}"

            f"{outcome:<20}"

            f"{item['standard_queries']:<8}"

            f"{item['adaptive_queries']:<8}"
        )

    print("=" * 110)


# ============================================================
# 9. EXPERIMENT SETTINGS
# ============================================================

NUM_SAMPLES = 10

MODEL_NAME = (
    "Arvan/Llama-3-3-70B-Instruct-R1AI"
)

RETRIEVER_NAME = "MedCPT"

CORPUS_NAME = "Textbooks"

K = 10

STANDARD_ROUNDS = 3

STANDARD_QUERIES_PER_ROUND = 2

ADAPTIVE_MAX_ITERATIONS = 3

ADAPTIVE_INITIAL_QUERIES = 2

CONFIDENCE_THRESHOLD = 85


# ============================================================
# 10. LOAD DATA
# ============================================================

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


# ============================================================
# 11. CREATE STANDARD MODEL
# ============================================================

standard_medrag = MedRAG(

    llm_name=MODEL_NAME,

    rag=True,

    follow_up=True,

    retriever_name=RETRIEVER_NAME,

    corpus_name=CORPUS_NAME,

    corpus_cache=True
)


# ============================================================
# 12. CREATE ADAPTIVE MODEL
# ============================================================

adaptive_medrag = AdaptiveMedRAG(

    llm_name=MODEL_NAME,

    rag=True,

    follow_up=True,

    retriever_name=RETRIEVER_NAME,

    corpus_name=CORPUS_NAME,

    corpus_cache=True
)


# ============================================================
# 13. RUN STANDARD
# ============================================================

# print("\n")
# print("=" * 110)
# print(
#     "RUNNING STANDARD i-MedRAG"
# )
# print("=" * 110)

# standard_result = run_standard_benchmark(

#     dataset=medqa_batch,

#     medrag_runner=standard_medrag,

#     fixed_rounds=STANDARD_ROUNDS,

#     n_queries=
#         STANDARD_QUERIES_PER_ROUND,

#     k=K
# )


# ============================================================
# 14. RUN CONFIDENCE ADAPTIVE
# ============================================================

print("\n")
print("=" * 110)
print(
    "RUNNING CONFIDENCE-ADAPTIVE i-MedRAG"
)
print("=" * 110)

adaptive_result = run_adaptive_benchmark(
    
    dataset=medqa_batch,

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

    query_budget_mode="confidence",

    max_iterations=
        ADAPTIVE_MAX_ITERATIONS,

    initial_queries=
        ADAPTIVE_INITIAL_QUERIES,

    confidence_threshold=
        CONFIDENCE_THRESHOLD,

    k=K
)


# ============================================================
# 15. QUESTION-LEVEL COMPARISON
# ============================================================

# question_comparison = (
#     build_question_comparison(

#         standard_result,

#         adaptive_result
#     )
# )


# ============================================================
# 16. SUMMARY COMPARISON
# ============================================================

# summary_comparison = (
#     build_summary_comparison(

#         standard_result,

#         adaptive_result
#     )
# )


# ============================================================
# 17. PRINT COMPARISON
# ============================================================

# print_summary_comparison(

#     standard_result,

#     adaptive_result
# )

# print_question_outcomes(
#     question_comparison
# )


# # ============================================================
# # 18. EXTRA COUNTS
# # ============================================================

# adaptive_improved = sum(
#     1
#     for item in question_comparison
#     if item["adaptive_improved"]
# )

# adaptive_hurt = sum(
#     1
#     for item in question_comparison
#     if item["adaptive_hurt"]
# )

# both_correct = sum(
#     1
#     for item in question_comparison
#     if item["both_correct"]
# )

# both_wrong = sum(
#     1
#     for item in question_comparison
#     if item["both_wrong"]
# )


# ============================================================
# 19. SAVE EVERYTHING
# ============================================================

combined_results = {

    # --------------------------------------------------------
    # Experiment information
    # --------------------------------------------------------

    "experiment": {

        "stage":
            "Stage 1",

        "purpose":
            (
                "Compare standard i-MedRAG against "
                "confidence-based adaptive i-MedRAG"
            ),

        "dataset":
            "GBaker/MedQA-USMLE-4-options",

        "split":
            "test",

        "num_questions":
            NUM_SAMPLES,

        "model":
            MODEL_NAME,

        "retriever":
            RETRIEVER_NAME,

        "corpus":
            CORPUS_NAME,

        "k":
            K,

        "standard": {

            "rounds":
                STANDARD_ROUNDS,

            "queries_per_round":
                STANDARD_QUERIES_PER_ROUND,

            "queries_per_question":
                (
                    STANDARD_ROUNDS
                    *
                    STANDARD_QUERIES_PER_ROUND
                )
        },

        "adaptive": {

            "max_iterations":
                ADAPTIVE_MAX_ITERATIONS,

            "initial_queries":
                ADAPTIVE_INITIAL_QUERIES,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "use_planner":
                False,

            "use_coverage":
                False,

            "use_novelty":
                False,

            "options_in_gate":
                True,

            "gate_features": [
                "confidence",
                "answerable",
                "discrimination"
            ]
        }
    },

    # --------------------------------------------------------
    # Full outputs
    # --------------------------------------------------------

    # "standard":
    #     standard_result,

    "confidence_adaptive":
        adaptive_result,

    # --------------------------------------------------------
    # Aggregate comparison
    # --------------------------------------------------------

    # "summary_comparison":
    #     summary_comparison,

    # # # --------------------------------------------------------
    # # # Question comparison
    # # # --------------------------------------------------------

    # "question_comparison":
    #     question_comparison,

    # # --------------------------------------------------------
    # # Outcome counts
    # # --------------------------------------------------------

    # "outcomes": {

    #     "adaptive_improved":
    #         adaptive_improved,

    #     "adaptive_hurt":
    #         adaptive_hurt,

    #     "both_correct":
    #         both_correct,

    #     "both_wrong":
    #         both_wrong
    # }
}


# ============================================================
# 20. SAVE JSON
# ============================================================

OUTPUT_FILE = (
    "stage1_standard_vs_confidence_20q_novelty.json"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        combined_results,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("FINAL STAGE 1 RESULT")
print("=" * 110)

# print(
#     f"Standard accuracy: "
#     f"{standard_result['accuracy']:.2f}%"
# )

print(
    f"Adaptive accuracy: "
    f"{adaptive_result['accuracy']:.2f}%"
)

print()

# print(
#     f"Standard avg queries: "
#     f"{standard_result['avg_queries']:.2f}"
# )

print(
    f"Adaptive avg queries: "
    f"{adaptive_result['avg_queries']:.2f}"
)

print()

# print(
#     f"Standard avg LLM calls: "
#     f"{standard_result['avg_llm_calls']:.2f}"
# )

print(
    f"Adaptive avg LLM calls: "
    f"{adaptive_result['avg_llm_calls']:.2f}"
)

print()

# print(
#     f"Standard avg total tokens: "
#     f"{standard_result['avg_total_tokens']:.2f}"
# )

print(
    f"Adaptive avg total tokens: "
    f"{adaptive_result['avg_total_tokens']:.2f}"
)

print()

# print(
#     f"Standard avg latency: "
#     f"{standard_result['avg_latency_seconds']:.2f}s"
# )

print(
    f"Adaptive avg latency: "
    f"{adaptive_result['avg_latency_seconds']:.2f}s"
)

print()

# print(
#     f"Adaptive improved: "
#     f"{adaptive_improved}"
# )

# print(
#     f"Adaptive hurt: "
#     f"{adaptive_hurt}"
# )

# print(
#     f"Both correct: "
#     f"{both_correct}"
# )

# print(
#     f"Both wrong: "
#     f"{both_wrong}"
# )

print()

print(
    f"Results saved to: "
    f"{OUTPUT_FILE}"
)

print("=" * 110)