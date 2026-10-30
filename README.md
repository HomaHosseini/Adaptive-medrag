### Efficient Iterative Retrieval for Medical Question Answering

This repository contains the implementation of an adaptive retrieval framework for medical question answering, developed as a bachelor's thesis in Computer Engineering at **Iran University of Science and Technology (IUST)**.

The project builds upon the iterative retrieval pipeline of [i-MedRAG](https://arxiv.org/abs/2408.00727) and investigates how retrieval depth can be dynamically adjusted according to question difficulty and the evidence accumulated during retrieval.

Inspired by [Stop-RAG](https://arxiv.org/abs/2510.14337), this project explores adaptive stopping strategies for medical RAG. It combines confidence-based assessment with a learned decision mechanism formulated as a finite-horizon Markov Decision Process (MDP). A Q-network trained using offline Q-learning with eligibility traces (Q(λ)) learns when to continue retrieval and when to stop and generate the final answer.

The objective is to reduce retrieval and generation costs while preserving medical question-answering performance.

## Approach

Given a medical question and its multiple-choice options, the system retrieves relevant passages from a medical textbook corpus using MedCPT. An LLM generates an intermediate answer and follow-up queries, which are used to retrieve additional evidence and refine the accumulated context.

After each retrieval round, the system evaluates whether the available evidence is sufficient to produce a final answer or whether another retrieval iteration is worthwhile.

Two adaptive stopping strategies are investigated:

- **Confidence-based stopping:** Uses LLM-generated confidence signals, including answerability and whether the available evidence sufficiently distinguishes between answer options. A deterministic decision rule uses these signals to determine whether retrieval should continue.
- **Learned adaptive stopping:** Formulates retrieval-depth selection as a finite-horizon MDP with two actions: STOP and CONTINUE. A neural Q-network estimates the expected value of each action from the current retrieval state, allowing the system to learn an adaptive retrieval policy.

The learned controller is trained offline using retrieval trajectories collected from the medical QA pipeline. Q(λ) learning is used to incorporate information from subsequent states into action-value estimation.

## Retrieval Workflow

1. Receive a medical question and its answer options.
2. Retrieve the top relevant passages using MedCPT.
3. Generate an intermediate answer and follow-up queries.
4. Retrieve additional evidence and update the retrieval state.
5. Evaluate the current state using the selected stopping strategy.
6. Either continue retrieval or stop and generate the final answer.

The system supports up to three retrieval rounds, with two follow-up queries generated per round.

## MDP Formulation

The adaptive retrieval process is modeled as a finite-horizon Markov Decision Process:

- **State:** Represents the question, answer options, and accumulated question–intermediate-answer evidence from previous retrieval rounds.
- **Actions:** STOP or CONTINUE.
- **Transition:** Continuing triggers another retrieval iteration and updates the accumulated evidence; stopping terminates retrieval.
- **Reward:** Balances answer quality against retrieval cost, encouraging useful additional retrieval while penalizing unnecessary computation.
- **Policy:** A learned policy derived from the Q-network selects the action with the highest estimated action value.

The Q-network uses a pretrained language-model encoder to represent the retrieval state, followed by neural layers that estimate action values. The controller is trained offline using Q-learning with eligibility traces (Q(λ)).

This formulation allows retrieval depth to be determined by the evolving evidence rather than a fixed retrieval budget.

## Experimental Setup

Experiments were conducted on **600 questions from the MedQA-USMLE dataset**, using medical textbooks as the retrieval corpus.

| Component                          | Configuration                    |
| ---------------------------------- | -------------------------------- |
| Dataset                            | MedQA-USMLE (4-option questions) |
| Evaluation samples                 | 600                              |
| Retriever                          | MedCPT                           |
| Retrieval corpus                   | Medical textbooks                |
| Generator                          | GPT-4o-mini                      |
| Retrieved passages                 | Top-10 per query                 |
| Maximum retrieval rounds           | 3                                |
| Follow-up queries                  | 2 per round                      |
| Maximum controller sequence length | 512 tokens                       |
| Controller training                | Offline Q-learning with Q(λ)     |

## Results

The adaptive approaches were evaluated against fixed-depth retrieval configurations, including one-round and three-round i-MedRAG.

| Method                    | Accuracy |
| ------------------------- | -------: |
| Fixed 1-round i-MedRAG    |   78.83% |
| Fixed 3-round i-MedRAG    |   82.17% |
| Confidence-based stopping |   80.67% |
| Learned adaptive stopping |   81.17% |

Compared with fixed three-round retrieval, the learned adaptive approach achieved:

| Metric           | Reduction |
| ---------------- | --------: |
| Token usage      |     60.0% |
| LLM calls        |     59.5% |
| Runtime          |     58.1% |
| Retrieval rounds |     59.2% |

The learned adaptive approach achieved an accuracy of **81.17%**, just 1.00 percentage point below fixed three-round retrieval, while reducing token usage by 60.0%, LLM calls by 59.5%, runtime by 58.1%, and retrieval rounds by 59.2%.

Paired statistical analyses, including Wilcoxon signed-rank and sign tests, were conducted to examine per-question performance differences. The reported tests showed statistically significant differences in the evaluated comparisons (p < 0.001).

Overall, the results indicate that the learned controller can substantially reduce computational overhead while retaining accuracy close to fixed three-round retrieval.

## Limitations and Future Work

The current experiments focus on a limited set of medical multiple-choice questions, one retrieval corpus, and a specific language-model configuration.

Potential directions for further research include:

- Evaluating on larger and more diverse medical QA benchmarks.
- Exploring alternative retrieval-state representations.
- Improving the generalization of learned stopping policies.
- Investigating different reward functions and retrieval-cost trade-offs.
- Evaluating the approach with additional language models and retrievers.
- Studying the relationship between evidence sufficiency, answer confidence, and retrieval depth.

## Repository Structure

The repository contains the implementation, controller training, and evaluation components. See the source directories for the corresponding modules and scripts.

```text
├── src/
│   ├── data/                   # Medical corpus loaders
│   ├── adapt_medrag.py         # Adaptive MedRAG pipeline
│   ├── medrag.py               # Core MedRAG implementation
│   ├── Q_network.py            # Q-network for adaptive stopping
│   └── ...
├── templates/                  # Prompt templates
├── stop_rag_ckpt/              # Trained controller checkpoints
├── mdp_dataset.py              # MDP dataset processing
├── mdp_dataset_collector.py    # Retrieval trajectory collection
├── requirements.txt
└── ...
```

## References

1. **Stop-RAG:** [Value-Based Retrieval Control for Iterative RAG](https://arxiv.org/abs/2510.14337).
2. **i-MedRAG:** [Iterative Retrieval-Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2408.00727).
3. Xiong et al. (2024). **Benchmarking Retrieval-Augmented Generation for Medicine.** Findings of ACL 2024. [Paper](https://aclanthology.org/2024.findings-acl.372/).

This project builds upon the [MedRAG repository](https://github.com/gzxiong/MedRAG). Please retain the original license and attribution notices when using or redistributing its code.
