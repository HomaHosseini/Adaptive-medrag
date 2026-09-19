import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("stop_rag_mdp_output/stop_rag_q_network.pt")
SCALER_PATH = Path("stop_rag_mdp_output/stop_rag_feature_scaler.json")
DATASET_PATH = Path(
    "stop_rag_mdp_output/stop_rag_mdp_training_dataset_70q.json"
)

THRESHOLD = 0.0


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# LOAD
# ============================================================

with open(SCALER_PATH, "r", encoding="utf-8") as f:
    scaler_data = json.load(f)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

feature_names = scaler_data["feature_names"]
mean = np.array(scaler_data["mean"], dtype=np.float32)
scale = np.array(scaler_data["scale"], dtype=np.float32)
medians = np.array(
    scaler_data["nan_fill_median"],
    dtype=np.float32
)

model = QNetwork(
    input_dim=len(feature_names),
    hidden=checkpoint["hidden"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def preprocess(features):

    x = np.array(features, dtype=np.float32)

    # Replace NaN / invalid values using training medians.
    invalid = ~np.isfinite(x)
    x[invalid] = medians[invalid]

    # Same StandardScaler transformation used during training.
    x = (x - mean) / scale

    return torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0)


# ============================================================
# PREDICT
# ============================================================

def predict_action(features):

    x = preprocess(features)

    with torch.no_grad():
        q = model(x)[0]

    q_stop = float(q[0])
    q_continue = float(q[1])

    margin = q_stop - q_continue

    if margin > THRESHOLD:
        action = "STOP"
    else:
        action = "CONTINUE"

    return {
        "q_stop": q_stop,
        "q_continue": q_continue,
        "margin": margin,
        "action": action,
    }


# ============================================================
# TEST ALL STATES
# ============================================================

correct = 0
total = 0

print("=" * 80)
print("STOP-RAG MODEL TEST")
print("=" * 80)

for record in dataset["records"]:

    result = predict_action(record["features"])

    print(
        f'{record["state_id"]:20s} '
        f'Q_stop={result["q_stop"]:.4f} '
        f'Q_cont={result["q_continue"]:.4f} '
        f'margin={result["margin"]:.4f} '
        f'→ {result["action"]}'
    )

    total += 1

print("\nTotal states:", total)