import json
import os
import time
import random

# IMPORTANT: load_dotenv() MUST run before importing anything that in turn
# imports medrag.py, since medrag.py reads OPENAI_API_KEY / OPENAI_API_BASE
# at MODULE IMPORT TIME (not inside a function). With load_dotenv() called
# later (as it was before), those os.getenv() calls saw an empty
# environment and silently fell back to config.py's defaults / the OpenAI
# SDK's real endpoint -- which is exactly why requests were hitting real
# OpenAI's servers instead of Chabokan.
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch

from datasets import load_dataset

from src.adapt_medrag import AdaptiveMedRAG
from mdp_dataset import (
    normalize_gold_answer,
    evaluate_stop_reward,
    make_json_safe
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Make CUDA operations as deterministic as possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For newer PyTorch versions
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass




# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 5

START_INDEX = 1

MODEL_NAME = (
    "OpenAI/meta-llama/llama-3.1-8b-instruct"
)

RETRIEVER_NAME = "MedCPT"

CORPUS_NAME = "Textbooks"

K = 10

N_ROUNDS = 3

N_QUERIES = 2

STOP_REWARD_TRIALS = 3

# Main output
OUTPUT_FILE = (
    "stage1_mdp_dataset_shuffled_10q_MedExpert.json"
)

# Checkpoint that is updated after EVERY question
CHECKPOINT_FILE = (
    "stage1_mdp_dataset_checkpoint.json"
)


# ============================================================
# LOAD MEDQA
# ============================================================

dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="test"
    )


dataset = dataset.shuffle(seed=SEED)
dataset = dataset.select(range(NUM_SAMPLES))


# ============================================================
# MODEL
# ============================================================

medrag = AdaptiveMedRAG(

    llm_name=MODEL_NAME,

    rag=True,

    follow_up=True,

    retriever_name=RETRIEVER_NAME,

    corpus_name=CORPUS_NAME,

    corpus_cache=True
)

# ------------------------------------------------------------
# gpt-4o-mini logprob-based reward scoring (via Arvan) was tried and parked:
# it required suppressing chain-of-thought reasoning to get clean
# single-token logprobs, which measurably hurt raw answer accuracy
# (forced immediate-answer vs. reason-then-answer is a known accuracy hit,
# independent of which model runs it). Reverted to the original Llama
# n_trials scoring below, now with an actual temperature fix -- see
# reward_temperature in evaluate_stop_reward. Kept here, commented, in
# case it's revisited later with a two-step (reason, then extract letter)
# version instead of forcing a bare-letter completion in one shot.
# ------------------------------------------------------------
# reward_medrag = AdaptiveMedRAG(
#     llm_name="Arvan/gpt-4o-mini",
#     rag=False,
# )


# ============================================================
# RECORDS
# ============================================================

question_records = []


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_checkpoint(records, filepath):
    """
    Save all completed questions immediately.
    This protects against network/API/computer failures.
    """

    output = {

        "experiment": {

            "name":
                "Stage 1 Stop-RAG-style MDP dataset",

            "dataset":
                "GBaker/MedQA-USMLE-4-options",

            "split":
                "train",

            "seed":
                SEED,

            "num_questions_completed":
                len(records),

            "model":
                MODEL_NAME,

            "corpus":
                CORPUS_NAME,

            "retriever":
                RETRIEVER_NAME,

            "k":
                K,

            "max_iterations":
                N_ROUNDS,

            "queries_per_iteration":
                N_QUERIES,

            "actions": [
                "STOP",
                "CONTINUE"
            ],

            "stop_reward_trials":
                STOP_REWARD_TRIALS
        },

        "questions":
            make_json_safe(records)
    }

    # Write to a temporary file first.
    # If something happens during writing, the previous
    # checkpoint is less likely to be corrupted.
    temp_file = filepath + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temp_file,
        filepath
    )


# ============================================================
# QUESTIONS
# ============================================================

for idx, item in enumerate(dataset):

    question_id = (
        f"MedQA_{START_INDEX + idx}"
    )

    question = item[
        "question"
    ]

    options = item[
        "options"
    ]

    gold = normalize_gold_answer(
        item["answer_idx"]
    )

    print("\n")
    print("=" * 100)
    print(
        f"QUESTION {idx + 1}/{NUM_SAMPLES}"
    )
    print(
        question_id
    )
    print("=" * 100)

    try:

        # ====================================================
        # FULL TRAJECTORY
        # ====================================================

        trajectory_result = medrag.collect_full_trajectory(

            question_id=question_id,

            question=question,

            options=options,

            k=K,

            n_rounds=N_ROUNDS,

            n_queries=N_QUERIES
        )

        trajectory = (
            trajectory_result[
                "trajectory"
            ]
        )

        # ====================================================
        # STOP REWARD
        # ====================================================

        
        stop_rewards = []

        for i, state in enumerate(trajectory):

            iteration = state["iteration"]

            # print("\n")
            # print(
            #     f"Evaluating STOP reward for S{iteration}"
            # )

            print(
                f"[DEBUG] Evaluating state: "
                f"{state['state_id']}"
            )

            # print(
            #     f"[DEBUG] Current context length: "
            #     f"{len(state['context'])}"
            # )
            # print(
            #     f"[STOP INPUT] {state['state_id']}"
            # )

            # print(
            #     f"Question chars: {len(question)}"
            # )

            # print(
            #     f"Context chars: {len(state['context'])}"
            # )

            # print(
            #     f"Context tail:\n{state['context'][-1000:]}"
            # )

            # ------------------------------------------------
            # Compare with next state
            # ------------------------------------------------

            if i < len(trajectory) - 1:

                next_state = trajectory[i + 1]

                # print(
                #     f"[DEBUG] Next state exists: "
                #     f"{next_state['state_id']}"
                # )

                # print(
                #     f"[DEBUG] Next context length: "
                #     f"{len(next_state['context'])}"
                # )

                # Show how much context was added
                current_context = state["context"]
                next_context = next_state["context"]

                if next_context.startswith(current_context):

                    new_context = next_context[
                        len(current_context):
                    ]

                    # print(
                    #     f"[DEBUG] New context added "
                    #     f"for {next_state['state_id']}: "
                    #     f"{len(new_context)} chars"
                    # )

                    # print(
                    #     f"[DEBUG] New context preview:\n"
                    #     f"{new_context[:1000]}"
                    # )

                else:

                    print(
                        "⚠️ WARNING: Next context does NOT "
                        "start with current context!"
                    )

            # ------------------------------------------------
            # STOP reward
            # ------------------------------------------------

            reward = evaluate_stop_reward(
                medrag_runner=medrag,
                question=question,
                options=options,
                gold=gold,
                state=state,
                n_trials=STOP_REWARD_TRIALS,
                reward_temperature=0.8,  # was silently 0.0 before -- see mdp_dataset.py fix
                reward_llm_runner=medrag,  # gpt-4o-mini logprob path parked, see note above
            )

            reward_record = {

                "state_id":
                    state["state_id"],

                "iteration":
                    state["iteration"],

                "reward":
                    reward["reward"],

                "n_trials":
                    reward["n_trials"],

                "trial_outputs":
                    reward.get("trial_outputs"),  # None in logprob mode -- see letter_probs/raw_answer instead

                "scoring_method":
                    reward.get("scoring_method", "n_trials"),

                "letter_probs":
                    reward.get("letter_probs"),
            }

            stop_rewards.append(
                reward_record
            )

            print(
                f"STOP reward: "
                f"{reward['reward']:.3f}"
            )

        # ====================================================
        # MDP TRANSITIONS
        # ====================================================

        transitions = []

        for i, state in enumerate(
            trajectory
        ):

            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            transitions.append({

                "state_id":
                    state["state_id"],

                "iteration":
                    state["iteration"],

                "action":
                    "STOP",

                "reward":
                    stop_rewards[i][
                        "reward"
                    ],

                "terminal":
                    True,

                "next_state_id":
                    None
            })

            # ------------------------------------------------
            # CONTINUE
            # ------------------------------------------------

            if i < len(
                trajectory
            ) - 1:

                transitions.append({

                    "state_id":
                        state["state_id"],

                    "iteration":
                        state["iteration"],

                    "action":
                        "CONTINUE",

                    "reward":
                        0.0,

                    "terminal":
                        False,

                    "next_state_id":
                        trajectory[
                            i + 1
                        ]["state_id"]
                })

        # ====================================================
        # QUESTION RECORD
        # ====================================================

        question_record = {

            "question_id":
                question_id,

            "question":
                question,

            "options":
                options,

            "gold":
                gold,

            # -----------------------------------------------
            # Full trajectory
            # -----------------------------------------------

            "trajectory":
                trajectory,

            # -----------------------------------------------
            # STOP reward
            # -----------------------------------------------

            "stop_rewards":
                stop_rewards,

            # -----------------------------------------------
            # MDP transitions
            # -----------------------------------------------

            "mdp_transitions":
                transitions,

            # -----------------------------------------------
            # Full trajectory result
            # -----------------------------------------------

            "full_trajectory_answer":
                trajectory_result[
                    "final_answer"
                ],

            "full_trajectory_metrics":
                trajectory_result[
                    "metrics"
                ]
        }

        # ====================================================
        # ADD SUCCESSFUL QUESTION
        # ====================================================

        question_records.append(
            question_record
        )

        # ====================================================
        # SAVE IMMEDIATELY
        # ====================================================

        save_checkpoint(
            question_records,
            CHECKPOINT_FILE
        )

        print("\n")
        print("-" * 100)
        print(
            f"✅ COMPLETED: {question_id}"
        )
        print(
            f"✅ Questions saved: "
            f"{len(question_records)}/{NUM_SAMPLES}"
        )
        print(
            f"💾 Checkpoint: "
            f"{CHECKPOINT_FILE}"
        )
        print("-" * 100)

    except Exception as e:

        # ====================================================
        # ERROR
        # ====================================================

        print("\n")
        print("!" * 100)
        print(
            f"❌ ERROR in {question_id}"
        )
        print(
            f"Error type: {type(e).__name__}"
        )
        print(
            f"Error message: {e}"
        )
        print("!" * 100)

        # Save everything completed BEFORE this question
        save_checkpoint(
            question_records,
            CHECKPOINT_FILE
        )

        print(
            f"💾 Previous results preserved."
        )

        print(
            f"💾 Completed questions: "
            f"{len(question_records)}"
        )

        # Continue with next question
        continue


# ============================================================
# FINAL OUTPUT
# ============================================================

output = {

    "experiment": {

        "name":
            "Stage 1 Stop-RAG-style MDP dataset",

        "dataset":
            "GBaker/MedQA-USMLE-4-options",

        "split":
            "train",

        "num_questions":
            len(question_records),

        "requested_num_questions":
            NUM_SAMPLES,

        "model":
            MODEL_NAME,

        "corpus":
            CORPUS_NAME,

        "retriever":
            RETRIEVER_NAME,

        "k":
            K,

        "max_iterations":
            N_ROUNDS,

        "queries_per_iteration":
            N_QUERIES,

        "actions": [
            "STOP",
            "CONTINUE"
        ],

        "stop_reward_trials":
            STOP_REWARD_TRIALS
    },

    "questions":
        make_json_safe(
            question_records
        )
}


# ============================================================
# SAVE FINAL FILE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("=" * 100)
print("MDP DATASET CREATED")
print("=" * 100)
print(
    f"Questions completed: "
    f"{len(question_records)}/{NUM_SAMPLES}"
)
print(
    f"Final output: "
    f"{OUTPUT_FILE}"
)
print(
    f"Checkpoint: "
    f"{CHECKPOINT_FILE}"
)
print("=" * 100)