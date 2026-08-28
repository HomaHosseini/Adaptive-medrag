from src.medrag import MedRAG
from src.adapt_medrag import AdaptiveMedRAG
import json
import os
import requests
import re, ast
from datasets import load_dataset




# ------------------------------------------------------------
# 1. Parse model answer
# ------------------------------------------------------------

def parse_predicted_letter(raw_answer):

    if raw_answer is None:
        return "UNKNOWN"

    text = str(raw_answer).strip()

    # Remove markdown code fences
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
    # JSON
    # --------------------------------------------------------

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):

            value = str(
                parsed.get("answer", "")
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
    # Python dict
    # --------------------------------------------------------

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, dict):

            value = str(
                parsed.get("answer", "")
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
    # Natural language
    # --------------------------------------------------------

    match = re.search(
        r"\b(?:answer|correct answer)\s*(?:is|:)\s*([A-F])\b",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # --------------------------------------------------------
    # A. option
    # --------------------------------------------------------

    match = re.search(
        r"(?m)^\s*([A-F])[\.\)\:]\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # --------------------------------------------------------
    # Just A
    # --------------------------------------------------------

    match = re.fullmatch(
        r"\s*([A-F])\s*",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return "UNKNOWN"


# ------------------------------------------------------------
# 2. Load the SAME 3 questions
# ------------------------------------------------------------

def load_medqa_subset(num_samples=3):

    print(
        f"Loading {num_samples} MedQA questions..."
    )

    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="test"
    )

    subset = dataset.select(
        range(num_samples)
    )

    return [
        {
            "id": f"MedQA_{i + 1}",
            "question": item["question"],
            "options": item["options"],
            "ground_truth": item["answer_idx"]
        }
        for i, item in enumerate(subset)
    ]

def run_standard_benchmark(

    dataset,

    medrag_runner,

    fixed_rounds=3,

    n_queries=2

):

    results = []

    correct = 0

    total_iterations = 0

    total_queries = 0

    print("\n")

    print("=" * 80)

    print("STANDARD i-MedRAG")

    print("=" * 80)

    for item in dataset:

        print("\n" + "-" * 80)

        print(

            f"Evaluating {item['id']} | STANDARD"

        )

        print("-" * 80)

        try:

            raw_answer, history = (

                medrag_runner.i_medrag_answer(

                    question=item["question"],

                    options=item["options"],

                    k=10,

                    n_rounds=fixed_rounds,

                    n_queries=n_queries

                )

            )

        except Exception as e:

            print(

                f"[ERROR] {item['id']}: {e}"

            )

            result = {

                "id": item["id"],

                "configuration": "standard",

                "predicted": "ERROR",

                "expected": item["ground_truth"],

                "correct": False,

                "iterations": fixed_rounds,

                "queries_generated":

                    fixed_rounds * n_queries,

                "error": str(e)

            }

            results.append(result)

            continue

        print("\nRAW FINAL ANSWER:")

        print(repr(raw_answer))

        predicted = parse_predicted_letter(

            raw_answer

        )

        is_correct = (

            predicted == item["ground_truth"]

        )

        if is_correct:

            correct += 1

        question_queries = (

            fixed_rounds * n_queries

        )

        total_iterations += fixed_rounds

        total_queries += question_queries

        result = {

            "id": item["id"],

            "configuration": "standard",

            "predicted": predicted,

            "expected": item["ground_truth"],

            "correct": is_correct,

            "iterations": fixed_rounds,

            "queries_generated":

                question_queries,

            "successful_retrievals":

                question_queries,

            "failed_retrievals": 0

        }

        results.append(result)

        print(

            f"\nPredicted: {predicted}"

        )

        print(

            f"Expected: {item['ground_truth']}"

        )

        print(

            f"Correct: {is_correct}"

        )

        print(

            f"Iterations: {fixed_rounds}"

        )

        print(

            f"Queries: {question_queries}"

        )


def run_adaptive_benchmark(
    dataset,
    medrag_runner,
    config_name,
    use_planner=False,
    use_planner_budget=False,
    use_coverage=False,
    use_novelty=False,
    max_iterations=3,
    initial_queries=2,
    confidence_threshold=85
):
    """
    Run one adaptive configuration on exactly the same dataset.

    Configuration examples:

    1. Confidence baseline:
        planner=False
        planner_budget=False
        coverage=False
        novelty=False

    2. Planner only:
        planner=True
        planner_budget=True
        coverage=False
        novelty=False

    3. Coverage:
        planner=True
        planner_budget=False
        coverage=True
        novelty=False

    4. Novelty:
        planner=False
        planner_budget=False
        coverage=False
        novelty=True

    5. Full:
        planner=True
        planner_budget=True
        coverage=True
        novelty=True
    """

    results = []

    correct = 0
    total_iterations = 0
    total_queries = 0
    total_successful_retrievals = 0
    total_failed_retrievals = 0

    print("\n")
    print("=" * 80)
    print(f"CONFIGURATION: {config_name}")
    print("=" * 80)

    for item in dataset:

        print("\n" + "-" * 80)
        print(f"Evaluating {item['id']} | {config_name}")
        print("-" * 80)

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
        k=10,
        max_iterations=max_iterations,
        initial_queries=initial_queries,
        confidence_threshold=confidence_threshold,
        use_planner=use_planner,
        use_planner_budget=use_planner_budget,
        use_coverage=use_coverage,
        use_novelty=use_novelty
    )

        print("\nRAW FINAL ANSWER:")
        print(repr(raw_answer))

        predicted = parse_predicted_letter(
            raw_answer
        )

        is_correct = (
            predicted == item["ground_truth"]
        )

        if is_correct:
            correct += 1

        total_iterations += iterations
        total_queries += queries_generated
        total_successful_retrievals += (
            successful_retrievals
        )
        total_failed_retrievals += (
            failed_retrievals
        )

        result = {

            "id": item["id"],

            "configuration": config_name,

            "predicted": predicted,

            "expected": item["ground_truth"],

            "correct": is_correct,

            "iterations": iterations,

            "confidence": confidence,

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

            # Keep these because they are useful
            # for later analysis.
            "evidence_history":
                evidence_history,

            "coverage_history":
                coverage_history
        }

        results.append(result)

        print(
            f"\nPredicted: {predicted}"
        )

        print(
            f"Expected: {item['ground_truth']}"
        )

        print(
            f"Correct: {is_correct}"
        )

        print(
            f"Iterations: {iterations}"
        )

        print(
            f"Queries generated: "
            f"{queries_generated}"
        )

        print(
            f"Successful retrievals: "
            f"{successful_retrievals}"
        )

        print(
            f"Failed retrievals: "
            f"{failed_retrievals}"
        )

        print(
            f"Budget history: "
            f"{query_budget_history}"
        )

        print(
            f"Final confidence: "
            f"{confidence:.1f}%"
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    n = len(dataset)

    accuracy = (
        correct / n * 100
        if n > 0
        else 0.0
    )

    avg_iterations = (
        total_iterations / n
        if n > 0
        else 0.0
    )

    avg_queries = (
        total_queries / n
        if n > 0
        else 0.0
    )

    avg_successful_retrievals = (
        total_successful_retrievals / n
        if n > 0
        else 0.0
    )

    avg_failed_retrievals = (
        total_failed_retrievals / n
        if n > 0
        else 0.0
    )

    summary = {

        "configuration": config_name,

        "accuracy": accuracy,

        "correct": correct,

        "total_questions": n,

        "avg_iterations":
            avg_iterations,

        "total_iterations":
            total_iterations,

        "avg_queries":
            avg_queries,

        "total_queries":
            total_queries,

        "avg_successful_retrievals":
            avg_successful_retrievals,

        "total_successful_retrievals":
            total_successful_retrievals,

        "avg_failed_retrievals":
            avg_failed_retrievals,

        "total_failed_retrievals":
            total_failed_retrievals,

        "results": results
    }

    print("\n")
    print("=" * 80)
    print(f"SUMMARY: {config_name}")
    print("=" * 80)

    print(
        f"Accuracy: "
        f"{accuracy:.2f}% "
        f"({correct}/{n})"
    )

    print(
        f"Average iterations: "
        f"{avg_iterations:.2f}"
    )

    print(
        f"Average queries: "
        f"{avg_queries:.2f}"
    )

    print(
        f"Average successful retrievals: "
        f"{avg_successful_retrievals:.2f}"
    )

    print(
        f"Average failed retrievals: "
        f"{avg_failed_retrievals:.2f}"
    )

    print("=" * 80)

    return summary


# ============================================================
# 1. Load ONLY questions 4, 5, 6
# ============================================================

all_questions = load_medqa_subset(
    num_samples=21
)



medqa_batch  = all_questions[20:]


print("\n")
print("=" * 80)
print("QUESTIONS USED")
print("=" * 80)

for item in medqa_batch:
    print(
        f"{item['id']} -> "
        f"Ground truth: {item['ground_truth']}"
    )


# ============================================================
# 2. Model
# ============================================================

MODEL_NAME = (
    "Arvan/Llama-3-3-70B-Instruct-R1AI"
)


# ============================================================
# 3. Create STANDARD model
# ============================================================

# standard_medrag = MedRAG(

#     llm_name=MODEL_NAME,

#     rag=True,

#     follow_up=True,

#     retriever_name="MedCPT",

#     corpus_name="Textbooks",

#     corpus_cache=True
# )


# ============================================================
# 4. Create ADAPTIVE model
# ============================================================

adaptive_medrag = AdaptiveMedRAG(

    llm_name=MODEL_NAME,

    rag=True,

    follow_up=True,

    retriever_name="MedCPT",

    corpus_name="Textbooks",

    corpus_cache=True
)


# ============================================================
# 5. Run STANDARD
# ============================================================

print("\n")
print("=" * 100)
print("RUNNING STANDARD i-MedRAG")
print("=" * 100)

# standard_result = run_standard_benchmark(

#     dataset=medqa_batch,

#     medrag_runner=standard_medrag,

#     fixed_rounds=3,

#     n_queries=2
# )


# ============================================================
# 6. Define adaptive experiments
# ============================================================

EXPERIMENTS = [

    # {
    #     "name": "adaptive_baseline",

    #     "use_planner": False,

    #     "use_planner_budget": False,

    #     "use_coverage": False,

    #     "use_novelty": False
    # },

    # {
    #     "name": "planner_only",

    #     "use_planner": True,

    #     "use_planner_budget": True,

    #     "use_coverage": False,

    #     "use_novelty": False
    # },

    {
        "name": "coverage",


        "use_planner": True,

        "use_planner_budget": False,

        "use_coverage": True,

        "use_novelty": False
    },

    # {
    #     "name": "novelty",

    #     "use_planner": False,

    #     "use_planner_budget": False,

    #     "use_coverage": False,

    #     "use_novelty": True
    # },

    # {
    #     "name": "full",

    #     "use_planner": True,

    #     "use_planner_budget": True,

    #     "use_coverage": True,

    #     "use_novelty": True
    # }
]


# ============================================================
# 7. Run ALL adaptive configurations
# ============================================================

all_adaptive_results = {}

for experiment in EXPERIMENTS:

    name = experiment["name"]

    print("\n")
    print("=" * 100)
    print(
        f"RUNNING ADAPTIVE CONFIGURATION: "
        f"{name}"
    )
    print("=" * 100)

    result = run_adaptive_benchmark(

        dataset=medqa_batch,

        medrag_runner=adaptive_medrag,

        config_name=name,

        use_planner=(
            experiment["use_planner"]
        ),

        use_planner_budget=(
            experiment["use_planner_budget"]
        ),

        use_coverage=(
            experiment["use_coverage"]
        ),

        use_novelty=(
            experiment["use_novelty"]
        ),

        max_iterations=3,

        initial_queries=2,

        confidence_threshold=85
    )

    all_adaptive_results[name] = result


# # ============================================================
# # 8. Combined comparison
# # ============================================================

# print("\n\n")
# print("=" * 110)
# print("STANDARD vs ADAPTIVE ABLATION — MedQA_4 to MedQA_6")
# print("=" * 110)

# print(
#     f"{'Configuration':<25}"
#     f"{'Accuracy':<12}"
#     f"{'Avg Iter':<12}"
#     f"{'Avg Queries':<14}"
#     f"{'Success Ret.':<16}"
#     f"{'Failed Ret.':<14}"
# )

# print("-" * 110)


# # ------------------------------------------------------------
# # Standard
# # ------------------------------------------------------------

# print(
#     f"{'standard':<25}"
#     f"{standard_result['accuracy']:<12.2f}"
#     f"{standard_result['avg_iterations']:<12.2f}"
#     f"{standard_result['avg_queries']:<14.2f}"
#     f"{standard_result.get('total_queries', 0) / len(medqa_batch):<16.2f}"
#     f"{0.00:<14.2f}"
# )


# # ------------------------------------------------------------
# # Adaptive
# # ------------------------------------------------------------

# for name, result in all_adaptive_results.items():

#     print(
#         f"{name:<25}"
#         f"{result['accuracy']:<12.2f}"
#         f"{result['avg_iterations']:<12.2f}"
#         f"{result['avg_queries']:<14.2f}"
#         f"{result['avg_successful_retrievals']:<16.2f}"
#         f"{result['avg_failed_retrievals']:<14.2f}"
#     )

# print("=" * 110)


# # ============================================================
# # 9. Compare query savings relative to STANDARD
# # ============================================================

# print("\n")
# print("=" * 80)
# print("QUERY SAVINGS vs STANDARD")
# print("=" * 80)

# standard_avg_queries = (
#     standard_result["avg_queries"]
# )

# for name, result in all_adaptive_results.items():

#     adaptive_avg_queries = (
#         result["avg_queries"]
#     )

#     if standard_avg_queries > 0:

#         query_savings = (
#             (
#                 standard_avg_queries
#                 - adaptive_avg_queries
#             )
#             / standard_avg_queries
#             * 100
#         )

#     else:
#         query_savings = 0.0

#     print(
#         f"{name:<25} "
#         f"{query_savings:>7.2f}% query reduction"
#     )

# print("=" * 80)


# # ============================================================
# # 10. Save STANDARD result
# # ============================================================

# with open(
#     "standard_medrag_q4_q6.json",
#     "w"
# ) as f:

#     json.dump(
#         standard_result,
#         f,
#         indent=4
#     )


# # ============================================================
# # 11. Save ADAPTIVE results
# # ============================================================

# with open(
#     "adaptive_ablation_q4_q6.json",
#     "w"
# ) as f:

#     json.dump(
#         all_adaptive_results,
#         f,
#         indent=4
#     )


# # ============================================================
# # 12. Save combined result
# # ============================================================

# combined_results = {

#     "questions": [
#         item["id"]
#         for item in medqa_batch
#     ],

#     "standard": standard_result,

#     "adaptive": all_adaptive_results
# }

# with open(
#     "medqa_q4_q6_all_results.json",
#     "w"
# ) as f:

#     json.dump(
#         combined_results,
#         f,
#         indent=4
#     )


# print("\n")
# print("=" * 80)
# print("RESULTS SAVED")
# print("=" * 80)
# print(
#     "standard_medrag_q4_q6.json"
# )
# print(
#     "adaptive_ablation_q4_q6.json"
# )
# print(
#     "medqa_q4_q6_all_results.json"
# )
# print("=" * 80)
