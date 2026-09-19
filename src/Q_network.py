import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, input_dim, hidden=64, dropout=0.10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.stop_head = nn.Linear(hidden, 1)
        self.cont_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)

        q_stop = self.stop_head(h).squeeze(-1)
        q_cont = self.cont_head(h).squeeze(-1)

        return torch.stack([q_stop, q_cont], dim=1)


class StopRAGController:

    def __init__(
        self,
        model_path,
        scaler_path,
        threshold=0.0,
        device=None,
    ):

        self.device = torch.device(
            device or
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # -------------------------
        # scaler
        # -------------------------

        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler = json.load(f)

        self.feature_names = scaler["feature_names"]

        self.mean = np.array(
            scaler["mean"],
            dtype=np.float32
        )

        self.scale = np.array(
            scaler["scale"],
            dtype=np.float32
        )

        self.medians = np.array(
            scaler["nan_fill_median"],
            dtype=np.float32
        )

        # -------------------------
        # model
        # -------------------------

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model = QNetwork(
            input_dim=checkpoint["input_dim"],
            hidden=checkpoint["hidden"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold

    # --------------------------------------------------------
    # Convert your trajectory state to feature vector
    # --------------------------------------------------------

    def extract_features(self, state):

        evidence = state.get("evidence_stats") or {}
        cost = state.get("cost") or {}

        def value(v):

            if v is None:
                return np.nan

            if isinstance(v, bool):
                return float(v)

            if isinstance(v, (int, float)):
                if np.isfinite(v):
                    return float(v)

            return np.nan

        features = {

            "iteration":
                state.get("iteration"),

            "confidence":
                state.get("confidence"),

            "answerable":
                state.get("answerable"),

            "discrimination":
                state.get("discrimination"),

            "num_documents":
                evidence.get("num_documents"),

            "num_new_documents":
                evidence.get("num_new_documents"),

            "document_novelty":
                evidence.get("document_novelty"),

            "semantic_novelty":
                evidence.get("semantic_novelty"),

            "semantic_median_novelty":
                evidence.get("semantic_median_novelty"),

            "high_semantic_novelty_fraction":
                evidence.get(
                    "high_semantic_novelty_fraction"
                ),

            "semantic_redundancy":
                evidence.get("semantic_redundancy"),

            "found_embeddings":
                evidence.get("found_embeddings"),

            "requested_query_count":
                state.get("requested_query_count"),

            "actual_query_count":
                state.get("actual_query_count"),

            "queries_so_far":
                cost.get("queries_so_far"),

            "llm_calls_so_far":
                cost.get("llm_calls_so_far"),

            "prompt_tokens_so_far":
                cost.get("prompt_tokens_so_far"),

            "completion_tokens_so_far":
                cost.get(
                    "completion_tokens_so_far"
                ),

            "total_tokens_so_far":
                cost.get("total_tokens_so_far"),

            "latency_seconds_so_far":
                cost.get(
                    "latency_seconds_so_far"
                ),

            "successful_retrievals_so_far":
                cost.get(
                    "successful_retrievals_so_far"
                ),

            "failed_retrievals_so_far":
                cost.get(
                    "failed_retrievals_so_far"
                ),
        }

        return np.array(
            [
                value(features[name])
                for name in self.feature_names
            ],
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(self, state):

        x = self.extract_features(state)

        # NaN replacement
        invalid = ~np.isfinite(x)

        x[invalid] = self.medians[invalid]

        # StandardScaler transformation
        x = (x - self.mean) / self.scale

        x = torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q = self.model(x)[0]

        q_stop = float(q[0])
        q_continue = float(q[1])

        margin = q_stop - q_continue

        if margin > self.threshold:
            action = "STOP"
        else:
            action = "CONTINUE"

        return {
            "action": action,
            "q_stop": q_stop,
            "q_continue": q_continue,
            "margin": margin,
        }