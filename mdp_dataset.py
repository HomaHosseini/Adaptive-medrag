# mdp_dataset.py

import ast
import json
import re
import time
from typing import Any, Dict, List, Optional


# ============================================================
# 1. PARSE MODEL ANSWER
# ============================================================

import re


def parse_predicted_letter(text):
    if not text:
        return "UNKNOWN"

    text = text.strip()
    match = re.search(
        r"##\s*Answer\s*\n\s*Answer\s*:\s*([A-D])\b",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()
    # --------------------------------------------------------
    # 1. Look specifically for the final answer section
    # --------------------------------------------------------
    answer_section = re.search(
        r"(?:##\s*Answer|###\s*Answer)\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if answer_section:
        final_text = answer_section.group(1).strip()

        # "The correct answer is: **B**"
        match = re.search(
            r"(?:correct\s+answer|best\s+answer|answer)"
            r"\s*(?:is|:)\s*\**\s*([A-D])\b",
            final_text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

        # "**B**"
        match = re.search(
            r"\*\*\s*([A-D])\s*\*\*",
            final_text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

        # "B. Complete failure..."
        match = re.search(
            r"^\s*([A-D])\s*[\.\):]",
            final_text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    # --------------------------------------------------------
    # 2. Search full response for explicit answer statements
    # --------------------------------------------------------
    matches = re.findall(
        r"(?:correct\s+answer|best\s+answer|final\s+answer|answer)"
        r"\s*(?:is|:)\s*\**\s*([A-D])\b",
        text,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1].upper()

    # --------------------------------------------------------
    # 3. Bolded answer like **B. text**
    # --------------------------------------------------------
    matches = re.findall(
        r"\*\*\s*([A-D])\s*[\.\):]",
        text,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1].upper()

    return "UNKNOWN"
# ============================================================
# 2. NORMALIZE GOLD ANSWER
# ============================================================

def normalize_gold_answer(answer_idx: Any) -> str:
    """
    Convert MedQA answer index to A/B/C/D.

    0 -> A
    1 -> B
    2 -> C
    3 -> D

    Also accepts an already-normalized letter.
    """

    if isinstance(answer_idx, int):

        if 0 <= answer_idx <= 3:
            return chr(
                ord("A") + answer_idx
            )

    text = str(answer_idx).strip().upper()

    if text in {
        "A",
        "B",
        "C",
        "D"
    }:
        return text

    return "UNKNOWN"


# ============================================================
# 3. JSON-SAFE CONVERSION
# ============================================================

def make_json_safe(obj: Any) -> Any:
    """
    Convert common Python / NumPy objects into values that
    json.dump() can serialize.
    """

    if obj is None:
        return None

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool
        )
    ):
        return obj

    if isinstance(obj, dict):

        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple)
    ):

        return [
            make_json_safe(value)
            for value in obj
        ]

    # NumPy scalar
    if hasattr(obj, "item"):

        try:
            return obj.item()

        except Exception:
            pass

    # NumPy array / tensor
    if hasattr(obj, "tolist"):

        try:
            return obj.tolist()

        except Exception:
            pass

    # Fallback
    return str(obj)

def evaluate_no_rag_state(
    medrag_runner,
    question,
    options,
    gold,
    n_trials=3
):
    """
    Evaluate the pre-retrieval S0 state using n independent
    LLM answer generations with no retrieved evidence.
    """

    trial_outputs = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency = 0.0

    for trial in range(1, n_trials + 1):

        messages = [
            {
                "role": "system",
                "content": medrag_runner.templates["i_medrag_system"]
            },
            {
                "role": "user",
                "content": medrag_runner.templates[
                    "follow_up_answer"
                ].format(
                    question=question,
                    options=options,
                    context=""
                )
            }
        ]

        start_time = time.time()

        raw_answer = medrag_runner.generate(messages)

        latency = time.time() - start_time

        predicted = parse_predicted_letter(raw_answer)

        correct = (
            predicted != "UNKNOWN"
            and predicted == gold
        )

        trial_outputs.append({
            "trial": trial,
            "raw_answer": raw_answer,
            "predicted": predicted,
            "correct": correct,
            "latency_seconds": latency
        })

        total_latency += latency

        # If your generate() exposes token metrics, add them here.
        # Otherwise leave them at 0 for now.

    reward = sum(
        x["correct"] for x in trial_outputs
    ) / n_trials

    return {
        "reward": reward,
        "n_trials": n_trials,
        "trial_outputs": trial_outputs,
        "cost": {
            "llm_calls": n_trials,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": total_latency
        }
    }


# ============================================================
# 4. COMPUTE COST-AWARE REWARD
# ============================================================

def compute_cost_aware_reward(
    accuracy_reward: float,
    queries_so_far: int,
    lambda_cost: float = 0.0
) -> float:
    """
    Optional cost-aware reward.

    R = accuracy - lambda * queries

    Keep lambda_cost=0.0 for the initial dataset if you
    don't want cost to affect the training target yet.
    """

    try:
        accuracy_reward = float(
            accuracy_reward
        )
    except (
        TypeError,
        ValueError
    ):
        accuracy_reward = 0.0

    try:
        queries_so_far = int(
            queries_so_far
        )
    except (
        TypeError,
        ValueError
    ):
        queries_so_far = 0

    return (
        accuracy_reward
        - lambda_cost * queries_so_far
    )


# ============================================================
# 5. EVALUATE STOP REWARD
# ============================================================

def evaluate_stop_reward(
    medrag_runner,
    question: str,
    options: Optional[Dict[str, str]],
    gold: str,
    state: Dict[str, Any],
    n_trials: int = 3,
    reward_llm_runner=None,
    reward_temperature: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate the reward of taking STOP at a particular state.

    IMPORTANT:
        - No retrieval is performed.
        - No new queries are generated.
        - The exact state context is reused.
        - Only final-answer generation is repeated.

    If `reward_llm_runner` is given (a MedRAG instance with
    llm_name="OpenAI/gpt-4o-mini" or similar), reward is computed from a
    SINGLE deterministic call's token logprobs on the gold letter --
    continuous 0-1, no sampling, no n_trials loop. This is both cheaper
    (1 call vs n_trials calls) and finer-grained than trial-averaging,
    since exact-match trial voting with the existing generate() (which
    runs greedy/temperature=0 decoding) gives no real variation to average
    over -- any fractional rewards you saw with n_trials averaging were
    incidental non-determinism in the serving stack, not real sampling.

    If `reward_llm_runner` is None (default), falls back to the original
    n_trials exact-match voting behavior, unchanged, using medrag_runner.

    The reward is:
        mean(correctness over trials)          [n_trials mode]
        P(gold letter | state)                 [reward_llm_runner mode]
    """

    # ========================================================
    # BUILD OPTION TEXT
    # ========================================================

    if options is not None:

        option_lines = []

        for key in sorted(
            options.keys()
        ):

            option_lines.append(
                f"{key}. {options[key]}"
            )

        options_str = "\n".join(
            option_lines
        )

    else:

        options_str = ""

    # ========================================================
    # QUESTION PROMPT
    # ========================================================

    question_prompt = (
        f"Here is the question:\n"
        f"{question}\n\n"
        f"{options_str}"
    )

    # ========================================================
    # STATE CONTEXT
    # ========================================================

    context = str(
        state.get(
            "context",
            ""
        )
    ).strip()

    # ========================================================
    # LOGPROB MODE (single call, continuous reward)
    # ========================================================

    if reward_llm_runner is not None:

        if options is None:
            raise ValueError(
                "reward_llm_runner (logprob-based reward) requires MCQ "
                "options -- it scores P(gold letter), which is undefined "
                "for open-ended questions."
            )

        option_letters = sorted(options.keys())
        is_local = not any(x in reward_llm_runner.llm_name.lower() for x in ("openai", "arvan", "gemini"))

        if is_local:
            # Reasoning-preserving path: SAME messages/template as the
            # n_trials branch below (full chain-of-thought allowed), so
            # there is no accuracy cost. The per-letter probability is read
            # off the model's own logits at wherever it actually commits to
            # its final answer, not forced onto the first token.
            messages = [
                {
                    "role": "system",
                    "content": medrag_runner.templates["i_medrag_system"],
                },
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"{question_prompt}\n\n"
                        f"{medrag_runner.templates['follow_up_answer']}"
                    ),
                },
            ]
            raw_text, letter_probs = reward_llm_runner.generate_answer_with_letter_probs_local(
                messages, option_letters=option_letters,
            )
            
        else:
            # API-based path (OpenAI-schema gateway, e.g. Liara/Chabokan, or
            # Arvan): now uses the SAME reasoning-permitting template as the
            # local path -- confirmed working via Chabokan-routed
            # Llama-3.1-8B-Instruct returning populated logprobs on normal
            # (non-bare-letter) completions. No accuracy cost vs. n_trials.
            messages = [
                {
                    "role": "system",
                    "content": medrag_runner.templates["i_medrag_system"],
                },
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"{question_prompt}\n\n"
                        f"{medrag_runner.templates['follow_up_answer']}"
                    ),
                },
            ]
            raw_text, letter_probs = reward_llm_runner.generate_answer_with_letter_probs(
                messages, option_letters=option_letters,
            )

       
        reward = letter_probs.get(gold, 0.0)

        return {
            "state_id": state.get("state_id"),
            "iteration": state.get("iteration"),
            "action": "STOP",
            "n_trials": 1,
            "reward": reward,
            "mean_accuracy": reward,
            "scoring_method": "logprob_local" if is_local else "logprob_api",
            "letter_probs": letter_probs,
            "raw_answer": raw_text,
            "queries_so_far": state.get("queries_so_far", None),
        }

    # ========================================================
    # N-TRIALS MODE (original behavior, unchanged)
    # ========================================================

    trial_outputs = []

    for trial_idx in range(
        n_trials
    ):

        trial_start = time.perf_counter()

        try:

            messages = [

                {
                    "role": "system",
                    "content":
                        medrag_runner.templates[
                            "i_medrag_system"
                        ]
                },

                {
                    "role": "user",
                    "content":
                        (
                            f"{context}\n\n"
                            f"{question_prompt}\n\n"
                            f"{medrag_runner.templates['follow_up_answer']}"
                        )
                }
            ]

            raw_answer = (
                medrag_runner.generate(
                    messages,
                    temperature=reward_temperature,  # was silently defaulting to 0.0 -- see note above
                    **kwargs
                )
            )

            predicted = (
                parse_predicted_letter(
                    raw_answer
                )
            )

            correct = (
                predicted != "UNKNOWN"
                and predicted == gold
            )

            trial_outputs.append({

                "trial":
                    trial_idx + 1,

                "raw_answer":
                    raw_answer,

                "predicted":
                    predicted,

                "correct":
                    correct,

                "latency_seconds":
                    time.perf_counter()
                    - trial_start
            })

        except Exception as e:

            trial_outputs.append({

                "trial":
                    trial_idx + 1,

                "raw_answer":
                    None,

                "predicted":
                    "ERROR",

                "correct":
                    False,

                "error":
                    str(e),

                "latency_seconds":
                    time.perf_counter()
                    - trial_start
            })

    # ========================================================
    # REWARD
    # ========================================================

    valid_trials = [
        item
        for item in trial_outputs
        if item["predicted"] != "ERROR"
    ]

    if valid_trials:

        reward = (
            sum(
                1.0
                if item["correct"]
                else 0.0
                for item in valid_trials
            )
            / len(valid_trials)
        )

    else:

        reward = 0.0

    # ========================================================
    # COST INFORMATION
    # ========================================================

    queries_so_far = state.get(
        "queries_so_far",
        None
    )

    result = {

        "state_id":
            state.get(
                "state_id"
            ),

        "iteration":
            state.get(
                "iteration"
            ),

        "action":
            "STOP",

        "n_trials":
            len(valid_trials),

        "reward":
            reward,

        "mean_accuracy":
            reward,

        "trial_outputs":
            trial_outputs,

        "queries_so_far":
            queries_so_far
    }

    # Optional reward variants
    result["cost_aware_reward_lambda_0.01"] = (
        compute_cost_aware_reward(
            accuracy_reward=reward,
            queries_so_far=(
                queries_so_far or 0
            ),
            lambda_cost=0.01
        )
    )

    result["cost_aware_reward_lambda_0.02"] = (
        compute_cost_aware_reward(
            accuracy_reward=reward,
            queries_so_far=(
                queries_so_far or 0
            ),
            lambda_cost=0.02
        )
    )

    result["cost_aware_reward_lambda_0.05"] = (
        compute_cost_aware_reward(
            accuracy_reward=reward,
            queries_so_far=(
                queries_so_far or 0
            ),
            lambda_cost=0.05
        )
    )

    return result


# ============================================================
# 6. BUILD MDP TRANSITIONS
# ============================================================

def build_mdp_transitions(
    states: List[Dict[str, Any]],
    stop_rewards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a full trajectory into Stage-1 MDP transitions.

    Action space:

        STOP
        CONTINUE

    Example:

        S1 --STOP------> terminal
        S1 --CONTINUE--> S2

        S2 --STOP------> terminal
        S2 --CONTINUE--> S3

        S3 --STOP------> terminal
    """

    transitions = []

    if not states:
        return transitions

    for index, state in enumerate(
        states
    ):

        state_id = state.get(
            "state_id"
        )

        iteration = state.get(
            "iteration"
        )

        # ====================================================
        # STOP
        # ====================================================

        stop_reward = 0.0

        if index < len(
            stop_rewards
        ):

            stop_reward = float(
                stop_rewards[index].get(
                    "reward",
                    0.0
                )
            )

        transitions.append({

            "state_id":
                state_id,

            "iteration":
                iteration,

            "action":
                "STOP",

            "reward":
                stop_reward,

            "terminal":
                True,

            "next_state_id":
                None
        })

        # ====================================================
        # CONTINUE
        # ====================================================

        if index < len(states) - 1:

            next_state_id = states[
                index + 1
            ].get(
                "state_id"
            )

            transitions.append({

                "state_id":
                    state_id,

                "iteration":
                    iteration,

                "action":
                    "CONTINUE",

                "reward":
                    0.0,

                "terminal":
                    False,

                "next_state_id":
                    next_state_id
            })

    return transitions


# ============================================================
# 7. ADD CUMULATIVE INFORMATION TO STATES
# ============================================================

def enrich_states_with_cumulative_cost(
    states: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Add cumulative query/retrieval counts to each state.

    This does not modify the original RAG logic.
    """

    cumulative_queries = 0
    cumulative_successful = 0
    cumulative_failed = 0

    enriched_states = []

    for state in states:

        current_queries = int(
            state.get(
                "actual_query_count",
                0
            )
            or 0
        )

        # If the collector did not store per-state query
        # count but has the query list, use that instead.
        if current_queries == 0:

            current_queries = len(
                state.get(
                    "queries",
                    []
                )
            )

        cumulative_queries += (
            current_queries
        )

        # ----------------------------------------------------
        # Retrieval counts
        # ----------------------------------------------------

        retrieval_records = state.get(
            "retrieval_records",
            []
        )

        current_successful = sum(
            1
            for item in retrieval_records
            if "error" not in item
        )

        current_failed = sum(
            1
            for item in retrieval_records
            if "error" in item
        )

        cumulative_successful += (
            current_successful
        )

        cumulative_failed += (
            current_failed
        )

        state_copy = dict(
            state
        )

        state_copy[
            "queries_so_far"
        ] = cumulative_queries

        state_copy[
            "successful_retrievals_so_far"
        ] = cumulative_successful

        state_copy[
            "failed_retrievals_so_far"
        ] = cumulative_failed

        # ----------------------------------------------------
        # Novelty availability
        # ----------------------------------------------------

        evidence_stats = (
            state_copy.get(
                "evidence_stats",
                {}
            )
            or {}
        )

        semantic_novelty = (
            evidence_stats.get(
                "semantic_novelty"
            )
        )

        state_copy[
            "novelty_available"
        ] = (
            semantic_novelty is not None
        )

        enriched_states.append(
            state_copy
        )

    return enriched_states


# ============================================================
# 8. BUILD ONE QUESTION RECORD
# ============================================================

def build_question_record(
    question_id: str,
    question: str,
    options: Dict[str, str],
    gold: str,
    states: List[Dict[str, Any]],
    stop_rewards: List[Dict[str, Any]],
    trajectory_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create the complete per-question MDP record.
    """

    # --------------------------------------------------------
    # Add cumulative cost/state information
    # --------------------------------------------------------

    states = enrich_states_with_cumulative_cost(
        states
    )

    # --------------------------------------------------------
    # Build transitions
    # --------------------------------------------------------

    transitions = build_mdp_transitions(
        states=states,
        stop_rewards=stop_rewards
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    if stop_rewards:

        final_state_reward = float(
            stop_rewards[-1].get(
                "reward",
                0.0
            )
        )

    else:

        final_state_reward = 0.0

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    record = {

        "question_id":
            question_id,

        "question":
            question,

        "options":
            make_json_safe(
                options
            ),

        "gold":
            gold,

        # ----------------------------------------------------
        # States
        # ----------------------------------------------------

        "trajectory":
            make_json_safe(
                states
            ),

        # ----------------------------------------------------
        # STOP rewards
        # ----------------------------------------------------

        "stop_rewards":
            make_json_safe(
                stop_rewards
            ),

        # ----------------------------------------------------
        # MDP transitions
        # ----------------------------------------------------

        "mdp_transitions":
            make_json_safe(
                transitions
            ),

        # ----------------------------------------------------
        # Terminal reward
        # ----------------------------------------------------

        "final_state_reward":
            final_state_reward
    }

    # --------------------------------------------------------
    # Full-trajectory generation result
    # --------------------------------------------------------

    if trajectory_result is not None:

        record[
            "full_trajectory_answer"
        ] = make_json_safe(
            trajectory_result.get(
                "final_answer"
            )
        )

        record[
            "full_trajectory_metrics"
        ] = make_json_safe(
            trajectory_result.get(
                "metrics",
                {}
            )
        )

    return record


# ============================================================
# 9. BUILD EXPERIMENT METADATA
# ============================================================

def build_experiment_config(
    dataset_name: str,
    split: str,
    num_questions: int,
    model_name: str,
    corpus_name: str,
    retriever_name: str,
    k: int,
    max_iterations: int,
    queries_per_iteration: int,
    stop_reward_trials: int,
) -> Dict[str, Any]:
    """
    Standard metadata block for the MDP dataset.
    """

    return {

        "stage":
            "Stage 1",

        "method":
            "Stop-RAG-style offline MDP",

        "dataset":
            dataset_name,

        "split":
            split,

        "num_questions":
            num_questions,

        "model":
            model_name,

        "corpus":
            corpus_name,

        "retriever":
            retriever_name,

        "retrieval_k":
            k,

        "max_iterations":
            max_iterations,

        "queries_per_iteration":
            queries_per_iteration,

        "action_space": [
            "STOP",
            "CONTINUE"
        ],

        "trajectory_collection":
            "Always continue until max_iterations",

        "reward_definition": {

            "stop":
                "Mean correctness over repeated final-answer trials",

            "continue":
                "Zero immediate reward; future reward comes from next state",

            "discount_factor":
                1.0
        },

        "stop_reward_trials":
            stop_reward_trials
    }


# ============================================================
# 10. SAVE DATASET
# ============================================================

def save_mdp_dataset(
    dataset_records: List[Dict[str, Any]],
    output_file: str,
    experiment_config: Dict[str, Any]
) -> None:
    """
    Save the complete offline MDP dataset.
    """

    output = {

        "experiment":
            make_json_safe(
                experiment_config
            ),

        "num_questions":
            len(dataset_records),

        "questions":
            make_json_safe(
                dataset_records
            )
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nMDP dataset saved to: "
        f"{output_file}"
    )


# ============================================================
# 11. BASIC DATASET VALIDATION
# ============================================================

def validate_mdp_dataset(
    dataset_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Basic sanity checks before training.

    Returns a validation report instead of raising errors
    for normal dataset problems.
    """

    report = {

        "num_questions":
            len(dataset_records),

        "questions_with_errors":
            0,

        "questions_without_trajectory":
            0,

        "total_states":
            0,

        "total_transitions":
            0,

        "states_with_missing_context":
            0,

        "states_with_missing_novelty":
            0
    }

    for record in dataset_records:

        if "error" in record:

            report[
                "questions_with_errors"
            ] += 1

        trajectory = record.get(
            "trajectory",
            []
        )

        if not trajectory:

            report[
                "questions_without_trajectory"
            ] += 1

        report[
            "total_states"
        ] += len(trajectory)

        transitions = record.get(
            "mdp_transitions",
            []
        )

        report[
            "total_transitions"
        ] += len(transitions)

        for state in trajectory:

            if not state.get(
                "context",
                ""
            ).strip():

                report[
                    "states_with_missing_context"
                ] += 1

            evidence_stats = state.get(
                "evidence_stats",
                {}
            ) or {}

            if evidence_stats.get(
                "semantic_novelty"
            ) is None:

                report[
                    "states_with_missing_novelty"
                ] += 1

    return report