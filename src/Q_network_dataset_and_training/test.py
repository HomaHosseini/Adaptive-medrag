"""
test_stop_rag_controller.py

Standalone smoke test for a trained Stop-RAG checkpoint. Loads
StopRAGController from train_stop_rag_qnetwork.py and walks real
trajectories from your dataset, printing Q(STOP) / Q(CONT) / margin /
decision at every iteration so you can eyeball whether the controller is
doing something sensible before wiring it into adapt_medrag.py.

Automatically detects the checkpoint's state_encoding (hierarchical vs.
concat) from its config.json and calls the controller the right way.

Also picks a mix of "easy" trajectories (reward is 1.0 at every iteration
-- for these, STOP vs CONTINUE genuinely doesn't matter, so don't read too
much into their margins) and "informative" ones (reward actually changes
across iterations -- these are what tell you if the model learned
anything useful).

No CLI -- edit the constants below and run it.
"""

import json
import torch

from Q_network_trainin import StopRAGController, get_doc_id

# ---- edit these ----
CHECKPOINT_DIR = "./stop_rag_ckpt"
DATA_FILE = "stage1_mdp_dataset_shuffled_20q.json"
N_EASY_EXAMPLES = 2
N_INFORMATIVE_EXAMPLES = 5
MARGIN_THRESHOLD = 0.0   # start at 0.0, tune later on a held-out val set
# ---------------------

# ---- edit these ----
CHECKPOINT_DIR = "src\\Q_network_dataset_and_training\\"
DATA_FILE = "src\\Q_network_dataset_and_training\\stage1_mdp_dataset_shuffled_20q.json"
N_EXAMPLES = 20
MARGIN_THRESHOLD = 0.0   # start at 0.0, tune later on a held-out val set
# ---------------------


def reward_is_informative(q: dict) -> bool:
    rewards = [sr["reward"] for sr in sorted(q["stop_rewards"], key=lambda sr: sr["iteration"])]
    return len(set(rewards)) > 1  # reward changes somewhere across iterations


def accumulate_documents_per_iteration(q: dict):
    """Mirrors the accumulation logic in load_trajectories(): returns a
    list (length T+1) of the deduplicated document list visible at each
    iteration, in the same {"id","title","content","retrieval_score"}
    shape your adapt_medrag.py pipeline's `accumulated_documents` uses."""
    states_by_iter = {s["iteration"]: s for s in q["trajectory"]}
    T = max(states_by_iter.keys())
    seen = {}
    out = []
    for t in range(T + 1):
        s = states_by_iter[t]
        for doc in s.get("iteration_documents", []):
            doc_id = get_doc_id(doc)
            seen[doc_id] = doc
        out.append(list(seen.values()))
    return out


def run_one(controller, q):
    print("=" * 100)
    print(f"[{q['question_id']}] gold={q['gold']}")
    print(q["question"][:200].replace("\n", " ") + ("..." if len(q["question"]) > 200 else ""))

    stop_rewards = {sr["iteration"]: sr["reward"] for sr in q["stop_rewards"]}
    docs_per_iter = accumulate_documents_per_iteration(q)

    for t, documents in enumerate(docs_per_iter):
        if controller.state_encoding == "hierarchical":
            q_stop, q_cont = controller.q_values(q["question"], q.get("options"), documents=documents)
        else:
            # legacy concat-mode checkpoint: fall back to the raw context string
            states_by_iter = {s["iteration"]: s for s in q["trajectory"]}
            context = states_by_iter[t].get("context", "")
            q_stop, q_cont = controller.q_values(q["question"], q.get("options"), context=context)

        margin = q_stop - q_cont
        decision = "STOP" if margin > controller.margin_threshold else "CONTINUE"
        r_stop = stop_rewards.get(t)
        print(
            f"  iter {t}: docs={len(documents):3d}  Q(STOP)={q_stop:+.3f}  Q(CONT)={q_cont:+.3f}  "
            f"margin={margin:+.3f}  -> {decision:8s}  (actual r(STOP) in data = {r_stop})"
        )

@torch.no_grad()
def diagnose_q_policy(
    model,
    trajectories,
    device,
    show_only_informative=True,
    margin_threshold=0.0,
):
    """
    Diagnostic evaluation for S0/S1/S2.

    Does NOT use confidence, LLM judge, or any extra signal.

    CONTINUE value is defined as the best reward available
    from later states in the offline trajectory.

    S3 is terminal and is not evaluated as a decision state.
    """

    model.eval()

    total = 0
    correct = 0
    total_regret = 0.0

    harmful_errors = 0
    harmless_ties = 0

    rows = []

    for traj in trajectories:

        T = len(traj.r_stop) - 1

        # S0 ... S(T-1) are decision states.
        # S_T is terminal.
        for t in range(T):

            if not traj.include_mask[t]:
                continue

            stop_value = float(
                traj.r_stop[t]
            )

            future_best = float(
                max(
                    traj.r_stop[t + 1:]
                )
            )

            # ------------------------------------------------
            # Optimal action
            # ------------------------------------------------

            if stop_value >= future_best:
                optimal_action = "STOP"
            else:
                optimal_action = "CONTINUE"

            # ------------------------------------------------
            # Model
            # ------------------------------------------------

            state_vector = (
                traj.state_vectors[t]
                .unsqueeze(0)
                .to(device)
            )

            q_stop, q_cont = model.heads(
                state_vector
            )

            q_stop = float(
                q_stop.item()
            )

            q_cont = float(
                q_cont.item()
            )

            margin = (
                q_stop - q_cont
            )

            predicted_action = (
                "STOP"
                if margin > margin_threshold
                else "CONTINUE"
            )

            # ------------------------------------------------
            # Regret
            # ------------------------------------------------

            if predicted_action == "STOP":

                achieved_value = stop_value

            else:

                achieved_value = future_best

            optimal_value = max(
                stop_value,
                future_best
            )

            regret = (
                optimal_value
                -
                achieved_value
            )

            # ------------------------------------------------
            # Stats
            # ------------------------------------------------

            is_correct = (
                predicted_action
                ==
                optimal_action
            )

            # If actions are tied in value, a different
            # action is harmless.
            harmless = (
                regret == 0.0
                and
                not is_correct
            )

            if is_correct:
                correct += 1

            elif harmless:
                harmless_ties += 1

            else:
                harmful_errors += 1

            total += 1
            total_regret += regret

            # ------------------------------------------------
            # Only show informative states if requested
            # ------------------------------------------------

            informative = (
                stop_value
                !=
                future_best
            )

            if (
                show_only_informative
                and not informative
            ):
                continue

            rows.append({
                "question_id":
                    traj.question_id,

                "state":
                    f"S{t}",

                "stop_reward":
                    stop_value,

                "future_best":
                    future_best,

                "optimal_action":
                    optimal_action,

                "q_stop":
                    q_stop,

                "q_continue":
                    q_cont,

                "margin":
                    margin,

                "predicted_action":
                    predicted_action,

                "regret":
                    regret,

                "harmful_error":
                    (
                        regret > 0
                    ),
            })

    print("\n" + "=" * 100)
    print("Q-POLICY DIAGNOSTIC")
    print("=" * 100)

    print(
        f"Decision states: {total}"
    )

    print(
        f"Exact action agreement: "
        f"{correct / max(1, total):.3f}"
    )

    print(
        f"Harmless ties: "
        f"{harmless_ties}"
    )

    print(
        f"Harmful errors: "
        f"{harmful_errors}"
    )

    print(
        f"Average regret: "
        f"{total_regret / max(1, total):.4f}"
    )

    print("=" * 100)

    for row in rows:

        print(
            f"{row['question_id']:12s} "
            f"{row['state']:3s} "
            f"rSTOP={row['stop_reward']:.3f} "
            f"future={row['future_best']:.3f} "
            f"optimal={row['optimal_action']:9s} "
            f"QSTOP={row['q_stop']:+.3f} "
            f"QCONT={row['q_continue']:+.3f} "
            f"margin={row['margin']:+.3f} "
            f"pred={row['predicted_action']:9s} "
            f"regret={row['regret']:.3f}"
        )

    return rows

def main():
    controller = StopRAGController(CHECKPOINT_DIR, margin_threshold=MARGIN_THRESHOLD)
    print(f"Loaded checkpoint with state_encoding={controller.state_encoding!r}")

    with open(DATA_FILE) as f:
        data = json.load(f)

    informative = [q for q in data["questions"] if reward_is_informative(q)]
    easy = [q for q in data["questions"] if not reward_is_informative(q)]

    print(f"\n{len(informative)}/{len(data['questions'])} questions have reward that "
          f"changes across iterations (the informative ones for judging the model).\n")

    for q in informative[:N_INFORMATIVE_EXAMPLES]:
        run_one(controller, q)

    if easy:
        print("\n" + "#" * 100)
        print("# 'Easy' examples below: reward is the same at every iteration, so STOP vs")
        print("# CONTINUE genuinely doesn't matter for them -- a flat/flipping margin near 0")
        print("# here is expected, not a sign the model is broken.")
        print("#" * 100)
        for q in easy[:N_EASY_EXAMPLES]:
            run_one(controller, q)


if __name__ == "__main__":
    main()