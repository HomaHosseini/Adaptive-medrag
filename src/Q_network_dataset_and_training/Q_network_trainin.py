"""
train_stop_rag_qnetwork.py

Trains a Stop-RAG-style value-based stopping controller for an iterative
MedRAG pipeline, following Park & Cho et al., "Stop-RAG: Value-Based
Retrieval Control for Iterative RAG" (NeurIPS 2025 workshop).

CHANGES vs. the original version
--------------------------------------------------------------------------
1. ENCODER BACKBONE -> MedCPT (instead of a generic DeBERTa encoder), to
   match your MedRAG retriever, which embeds both the corpus and queries
   with `ncbi/MedCPT-Query-Encoder` using CLS-token pooling (see
   `CustomizeSentenceTransformer` in utils.py). We do the same here.

2. NO CLI. All configuration lives in the `Config` dataclass below. Edit
   the fields directly (or import this file and construct/modify a
   `Config()` yourself) and call `train(cfg)`.

3. STATE ENCODING -> hierarchical, per-document (default), instead of one
   giant truncated string.
   MedCPT is a standard BERT-base model with a hard 512-token position
   limit. Your accumulated `context` field only grows (new retrieval
   results get appended) and can reach several thousand tokens by
   iteration 3, which is far beyond that limit. Cramming it all into one
   string and truncating -- even truncating smartly, keeping the most
   recent tokens -- still permanently throws away older evidence once a
   trajectory runs long enough, and every trajectory eventually runs into
   this.

   Your dataset actually gives you something better to work with: each
   state's `iteration_documents` (and running `accumulated_document_ids`)
   are individual documents with `title` / `content` / `retrieval_score`,
   each only ~80-250 tokens -- comfortably inside MedCPT's budget on its
   own. So instead of one huge sequence, we now:
     (a) encode the question once with MedCPT,
     (b) encode EVERY accumulated document individually with MedCPT
         (title+content pair, exactly the [title, content] format your
         own utils.py already feeds to MedCPT for corpus indexing),
     (c) pool the per-document embeddings into one fixed-size evidence
         vector (score-weighted average by default, using the retriever's
         own `retrieval_score` as the weight -- or plain mean, see
         `Config.evidence_pooling`),
     (d) concatenate [question_embedding, evidence_embedding] and feed
         that into the STOP / CONTINUE heads.
   Because pooling doesn't care how many documents there are, no evidence
   is ever dropped due to length, no matter how many iterations run. As a
   side benefit this is also cheaper to train: most of a trajectory's
   documents are already-seen repeats across iterations (retrieval keeps
   re-surfacing the same top hits), so each unique document is encoded
   ONCE per trajectory and its embedding is reused across every state that
   includes it, rather than re-encoding a growing multi-thousand-token
   string at every single iteration.

   The old behaviour (encode question+options+context as one truncated
   string) is still available via `Config.state_encoding = "concat"` if
   you want to compare the two, e.g. for an ablation. In that mode, text_b
   (the evidence) is truncated from the LEFT so the TAIL -- the most
   recently retrieved evidence -- is what's kept, and the question is
   never truncated. That's still lossy for long trajectories; "hierarchical"
   is the one that doesn't lose anything.

4. STILL NO confidence / answerable / discrimination / LLM-judge signal.
   The state representation is built ONLY from question + options +
   accumulated retrieved documents, and rewards are read directly from
   `stop_rewards` (F1/exact-match-style scores against the gold answer,
   not an LLM self-assessment). Nothing in this file reads `confidence`,
   `answerable`, `discrimination`, or `assessment_reasoning`.

DATA FORMAT EXPECTED
--------------------------------------------------------------------------
One or more JSON files shaped like your `stage1_mdp_dataset_*.json`, where
each trajectory state has `context` (accumulated evidence string) AND
`iteration_documents` (list of {"id"/"doc_id", "title", "content",
"retrieval_score"} for that iteration's retrieved docs) plus
`accumulated_document_ids` (running dedup'd id list). See section 2 below.

NOTE ON REWARDS
--------------------------------------------------------------------------
We do NOT read `mdp_transitions[*].reward` for the CONTINUE action (it's
always 0.0 in your files, even at the terminal decision, which would be
wrong under eq. (14) of the paper). Instead we compute everything directly
from `stop_rewards`, which you have for every iteration including the
terminal one:

    r(s_t, STOP)     = stop_rewards[t].reward                (eq. 13)
    r(s_t, CONTINUE) = 0                        if t <  T-1   (eq. 14)
                      = stop_rewards[T].reward   if t == T-1   (eq. 14)

USAGE (no CLI -- just run it, or import it)
--------------------------------------------------------------------------
    python train_stop_rag_qnetwork.py

...or from a notebook / another script:

    from train_stop_rag_qnetwork import Config, train
    cfg = Config(data=["my_file.json"], epochs=10)
    train(cfg)

Then use `StopRAGController` (bottom of this file) at inference time inside
your MedRAG loop instead of `choose_query_budget` / `check_stopping_gate`.
"""

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


# ============================================================
# 0. CONFIG  (edit these fields directly -- no CLI)
# ============================================================

@dataclass
class Config:
    # ---- data ----
    data: List[str] = field(default_factory=lambda: [
        "stage1_mdp_dataset_shuffled_20q.json",
        "stage1_mdp_dataset_shuffled_20_70q.json",
    ])
    val_frac: float = 0.15
    no_filter: bool = False  # if True, disables the paper's step-4 filtering

    # ---- encoder backbone ----
    # "query"   -> ncbi/MedCPT-Query-Encoder   (matches your Retriever's default)
    # "article" -> ncbi/MedCPT-Article-Encoder (trained on longer title+abstract
    #              pairs; a reasonable alternative for the per-document encoding)
    encoder_variant: str = "query"

    # ---- state representation ----
    # "hierarchical" (default, recommended): encode question + every
    #     accumulated document individually, pool -- no data loss.
    # "concat" (legacy/ablation): one string, tail-truncated to max_length.
    state_encoding: str = "hierarchical"
    # only used when state_encoding == "hierarchical":
    #   "score_weighted" -> softmax over each doc's `retrieval_score`
    #   "mean"            -> plain average
    evidence_pooling: str = "score_weighted"

    max_length: int = 512  # per encoder call; docs/questions are short enough
                            # in "hierarchical" mode that this is rarely hit
    head_hidden_size: int = 512
    dropout: float = 0.1

    # ---- training ----
    output_dir: str = "./stop_rag_ckpt"
    epochs: int = 20
    batch_size: int = 2  # trajectories per batch (not states) -- lower this
                          # first if you still OOM; in "hierarchical" mode a
                          # single trajectory can flatten to 20-40+ document
                          # encoder calls, so the *effective* batch the GPU
                          # sees is much bigger than this number suggests.
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lambda_start: float = 1.0
    lambda_end: float = 0.1
    seed: int = 42

    # ---- memory ----
    use_amp: bool = True             # mixed precision (fp16 autocast + GradScaler);
                                      # roughly halves activation memory on CUDA
    gradient_checkpointing: bool = True  # trade compute for memory in the encoder's
                                          # backward pass -- helps a lot on small GPUs


ENCODER_NAMES = {
    "query": "ncbi/MedCPT-Query-Encoder",
    "article": "ncbi/MedCPT-Article-Encoder",
}


# ============================================================
# 1. STATE TEXT CONSTRUCTION
#    -- still no confidence / answerable / discrimination fields
# ============================================================

def build_question_text(question: str, options: Optional[dict]) -> str:
    """Question + options only, no evidence. Used as one half of every
    state representation in both encoding modes."""
    parts = [f"Question: {question.strip()}"]
    if options:
        opt_str = " ".join(f"({k}) {v}" for k, v in sorted(options.items()))
        parts.append(f"Options: {opt_str}")
    return " ".join(parts)


def build_state_pair(question: str, options: Optional[dict], context: str) -> Tuple[str, str]:
    """(text_a, text_b) pair for "concat" mode: question+options, and the
    accumulated evidence string."""
    text_a = build_question_text(question, options)
    ctx = (context or "").strip()
    text_b = ctx if ctx else "(no evidence retrieved yet)"
    return text_a, text_b


def get_doc_id(doc: dict) -> str:
    """Mirrors the id-resolution convention already used in adapt_medrag.py
    (`doc.get("id") or doc.get("doc_id")`, falling back to a content hash)."""
    doc_id = doc.get("id") or doc.get("doc_id")
    if doc_id is not None:
        return str(doc_id)
    return "hash:" + str(hash(doc.get("content", "")))


# ============================================================
# 2. OFFLINE DATASET CONSTRUCTION
# ============================================================

@dataclass
class Trajectory:
    question_id: str
    question_text: str
    # "concat" mode:
    pairs: List[Tuple[str, str]] = field(default_factory=list)       # length T+1
    # "hierarchical" mode:
    unique_docs: Dict[str, dict] = field(default_factory=dict)       # doc_id -> {"title","content","retrieval_score"}
    state_doc_ids: List[List[str]] = field(default_factory=list)     # length T+1
    # shared:
    r_stop: List[float] = field(default_factory=list)                # length T+1
    include_mask: List[bool] = field(default_factory=list)           # length T


def _state_reward_lookup(question_entry: dict) -> List[float]:
    """Aligns stop_rewards to iteration index -> list of length T+1."""
    rewards_by_iter = {}
    for sr in question_entry["stop_rewards"]:
        rewards_by_iter[sr["iteration"]] = float(sr["reward"])

    max_iter = max(rewards_by_iter.keys())
    return [rewards_by_iter[i] for i in range(max_iter + 1)]


def load_trajectories(paths: List[str], filter_uninformative: bool = True) -> List[Trajectory]:
    """
    Loads and merges one or more dataset files into a flat list of
    Trajectory objects, one per question. Builds BOTH the "concat" pairs
    and the "hierarchical" per-document structures, so you can switch
    `Config.state_encoding` without reloading data.
    """
    trajectories: List[Trajectory] = []

    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        for q in data["questions"]:
            states_by_iter = {s["iteration"]: s for s in q["trajectory"]}
            T = max(states_by_iter.keys())  # terminal state index

            question_text = build_question_text(q["question"], q.get("options"))

            pairs = []
            unique_docs: Dict[str, dict] = {}
            state_doc_ids: List[List[str]] = []

            for t in range(T + 1):
                s = states_by_iter[t]

                # concat-mode pair
                pairs.append(build_state_pair(q["question"], q.get("options"), s.get("context", "")))

                # hierarchical-mode accumulation
                for doc in s.get("iteration_documents", []):
                    doc_id = get_doc_id(doc)
                    unique_docs[doc_id] = {
                        "title": doc.get("title", ""),
                        "content": doc.get("content", ""),
                        "retrieval_score": float(doc.get("retrieval_score", 0.0)),
                    }
                # snapshot of doc ids visible AT this state (cheap: list of str refs)
                state_doc_ids.append(list(unique_docs.keys()))

            r_stop = _state_reward_lookup(q)
            if len(r_stop) != T + 1:
                # Be defensive: pad/truncate rather than crash on odd files.
                r_stop = (r_stop + [r_stop[-1]] * (T + 1))[: T + 1]

            traj = Trajectory(
                question_id=q["question_id"],
                question_text=question_text,
                pairs=pairs,
                unique_docs=unique_docs,
                state_doc_ids=state_doc_ids,
                r_stop=r_stop,
            )

            # Precompute which decision states (t = 0..T-1) are worth training on.
            # Filter rule from the paper (step 4 of "Offline dataset construction"):
            #   drop states where r(st, STOP) == 0 AND Qhat^(T-t)(st, CONT) == 0
            include_mask = []
            for t in range(T):
                r_stop_t = r_stop[t]
                future_stop_rewards = r_stop[t + 1: T]
                r_continue_last = r_stop[T]
                q_full_unroll = max(future_stop_rewards + [r_continue_last])

                keep = True
                if filter_uninformative and r_stop_t == 0.0 and q_full_unroll == 0.0:
                    keep = False
                include_mask.append(keep)

            traj.include_mask = include_mask
            trajectories.append(traj)

    return trajectories


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


def collate_trajectories(batch: List[Trajectory]) -> List[Trajectory]:
    return batch


# ============================================================
# 3. Q-NETWORK  (MedCPT encoder + two heads: STOP, CONTINUE)
# ============================================================

class StopRAGQNetwork(nn.Module):
    def __init__(self, model_name: str, hidden_size: int = 512, dropout: float = 0.1,
                 pooling: str = "cls", state_encoding: str = "hierarchical",
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.enc_dim = self.encoder.config.hidden_size
        self.pooling = pooling  # "cls" matches MedCPT's own training/inference convention
        self.state_encoding = state_encoding

        if gradient_checkpointing:
            # Trades compute for memory: activations aren't kept for the
            # backward pass, they're recomputed. Big memory win on small
            # GPUs (e.g. Colab's T4) at some training-speed cost.
            self.encoder.gradient_checkpointing_enable()
            self.encoder.config.use_cache = False

        head_input_dim = self.enc_dim * 2 if state_encoding == "hierarchical" else self.enc_dim

        def make_head():
            return nn.Sequential(
                nn.Linear(head_input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        self.stop_head = make_head()
        self.cont_head = make_head()

        if state_encoding == "hierarchical":
            # Learned representation for "no evidence retrieved yet" (state t=0).
            self.no_evidence_embedding = nn.Parameter(torch.zeros(self.enc_dim))

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        """Runs the encoder + pooling only. Returns (N, enc_dim)."""
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return self._pool(out.last_hidden_state, attention_mask)

    def heads(self, pooled):
        """Runs the STOP/CONTINUE heads on already-pooled state vectors."""
        q_stop = self.stop_head(pooled).squeeze(-1)
        q_cont = self.cont_head(pooled).squeeze(-1)
        return q_stop, q_cont

    def forward_concat(self, input_ids, attention_mask, token_type_ids=None):
        """"concat" mode: one encoder call per state, straight into the heads."""
        pooled = self.encode(input_ids, attention_mask, token_type_ids)
        return self.heads(pooled)


def encode_pairs(tokenizer, pairs: List[Tuple[str, str]], max_length: int, device):
    """Tokenizes a list of (text_a, text_b) pairs as BERT-style pair inputs:
    [CLS] text_a [SEP] text_b [SEP].

    Custom truncation instead of the tokenizer's default: text_a is kept
    intact whenever possible, and text_b is truncated from the LEFT so its
    TAIL survives. This matters for "concat" mode, where text_b is the
    accumulated (growing) evidence string and the tail is the most
    recently retrieved evidence. In "hierarchical" mode this function is
    called on (title, content) doc pairs and (question, "") pairs, which
    are almost always short enough that truncation never triggers.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    special_overhead = 3  # [CLS] text_a [SEP] text_b [SEP]

    all_input_ids = []
    all_token_type_ids = []

    for text_a, text_b in pairs:
        ids_a = tokenizer.encode(text_a, add_special_tokens=False)
        budget_for_pair = max_length - special_overhead
        if len(ids_a) > budget_for_pair:
            ids_a = ids_a[:budget_for_pair]  # last-resort clip; keeps the start

        budget_for_b = max(0, budget_for_pair - len(ids_a))
        ids_b = tokenizer.encode(text_b, add_special_tokens=False)
        if len(ids_b) > budget_for_b:
            ids_b = ids_b[len(ids_b) - budget_for_b:]  # keep the TAIL

        input_ids = [cls_id] + ids_a + [sep_id] + ids_b + [sep_id]
        token_type_ids = [0] * (2 + len(ids_a)) + [1] * (1 + len(ids_b))

        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)

    max_len_in_batch = max(len(ids) for ids in all_input_ids)
    input_ids_t = torch.full((len(pairs), max_len_in_batch), pad_id, dtype=torch.long)
    attn_mask_t = torch.zeros((len(pairs), max_len_in_batch), dtype=torch.long)
    type_ids_t = torch.zeros((len(pairs), max_len_in_batch), dtype=torch.long)

    for i, (ids, types) in enumerate(zip(all_input_ids, all_token_type_ids)):
        input_ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn_mask_t[i, :len(ids)] = 1
        type_ids_t[i, :len(types)] = torch.tensor(types, dtype=torch.long)

    return {
        "input_ids": input_ids_t.to(device),
        "attention_mask": attn_mask_t.to(device),
        "token_type_ids": type_ids_t.to(device),
    }


def pool_evidence(doc_embeddings: torch.Tensor, scores: List[float], pooling: str, device) -> torch.Tensor:
    """doc_embeddings: (K, enc_dim). Returns (enc_dim,)."""
    if pooling == "score_weighted":
        score_t = torch.tensor(scores, device=device, dtype=doc_embeddings.dtype)
        weights = torch.softmax(score_t, dim=0)
    else:  # "mean"
        k = doc_embeddings.shape[0]
        weights = torch.full((k,), 1.0 / k, device=device, dtype=doc_embeddings.dtype)
    return (doc_embeddings * weights.unsqueeze(-1)).sum(dim=0)


def run_hierarchical_batch(model: StopRAGQNetwork, tokenizer, batch: List[Trajectory],
                            cfg: Config, device):
    """
    Encodes every unique question and every unique document ONCE per
    trajectory (not once per state -- documents repeat across iterations
    since retrieval keeps re-surfacing top hits), then builds each state's
    [question_embedding, pooled_evidence_embedding] vector and runs the
    heads on all states in the batch in one go.

    Returns: per_traj_stop, per_traj_cont -- lists of (T_i+1,) tensors,
    one pair per trajectory in `batch`, aligned with `traj.r_stop`.
    """
    # ---- flatten: one question pair + one pair per unique doc, per trajectory ----
    flat_pairs: List[Tuple[str, str]] = []
    owner: List[Tuple[int, str, Optional[str]]] = []  # (batch_idx, "question"|"doc", doc_id or None)

    for bi, traj in enumerate(batch):
        flat_pairs.append((traj.question_text, ""))
        owner.append((bi, "question", None))
        for doc_id, doc in traj.unique_docs.items():
            flat_pairs.append((doc.get("title", ""), doc.get("content", "")))
            owner.append((bi, "doc", doc_id))

    enc = encode_pairs(tokenizer, flat_pairs, cfg.max_length, device)
    pooled = model.encode(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))

    question_pooled: Dict[int, torch.Tensor] = {}
    doc_pooled: Dict[Tuple[int, str], torch.Tensor] = {}
    for i, (bi, kind, key) in enumerate(owner):
        if kind == "question":
            question_pooled[bi] = pooled[i]
        else:
            doc_pooled[(bi, key)] = pooled[i]

    # ---- build per-state [question, evidence] vectors ----
    all_state_vecs = []
    traj_lengths = []
    for bi, traj in enumerate(batch):
        qvec = question_pooled[bi]
        state_vecs = []
        for doc_ids in traj.state_doc_ids:
            if not doc_ids:
                evid_vec = model.no_evidence_embedding
            else:
                embs = torch.stack([doc_pooled[(bi, did)] for did in doc_ids], dim=0)
                scores = [traj.unique_docs[did].get("retrieval_score", 0.0) for did in doc_ids]
                evid_vec = pool_evidence(embs, scores, cfg.evidence_pooling, device)
            state_vecs.append(torch.cat([qvec, evid_vec], dim=-1))
        stacked = torch.stack(state_vecs, dim=0)  # (T_i+1, 2*enc_dim)
        all_state_vecs.append(stacked)
        traj_lengths.append(stacked.shape[0])

    all_state_vecs = torch.cat(all_state_vecs, dim=0)  # (sum(T_i+1), 2*enc_dim)
    q_stop_flat, q_cont_flat = model.heads(all_state_vecs)

    per_traj_stop, per_traj_cont = [], []
    cursor = 0
    for n in traj_lengths:
        per_traj_stop.append(q_stop_flat[cursor:cursor + n])
        per_traj_cont.append(q_cont_flat[cursor:cursor + n])
        cursor += n

    return per_traj_stop, per_traj_cont


def run_concat_batch(model: StopRAGQNetwork, tokenizer, batch: List[Trajectory],
                      cfg: Config, device):
    """Legacy path: one (question+options, evidence) string pair per state."""
    all_pairs = []
    for traj in batch:
        all_pairs.extend(traj.pairs)

    enc = encode_pairs(tokenizer, all_pairs, cfg.max_length, device)
    q_stop_flat, q_cont_flat = model.forward_concat(
        enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))

    per_traj_stop, per_traj_cont = [], []
    cursor = 0
    for traj in batch:
        n = len(traj.pairs)
        per_traj_stop.append(q_stop_flat[cursor:cursor + n])
        per_traj_cont.append(q_cont_flat[cursor:cursor + n])
        cursor += n
    return per_traj_stop, per_traj_cont


def run_batch(model, tokenizer, batch, cfg, device):
    if cfg.state_encoding == "hierarchical":
        return run_hierarchical_batch(model, tokenizer, batch, cfg, device)
    return run_concat_batch(model, tokenizer, batch, cfg, device)


# ============================================================
# 4. Q(lambda) TARGET COMPUTATION  (full-width forward-view, eqs. 6-12)
# ============================================================

def compute_qlambda_targets_for_trajectory(
    r_stop: List[float],
    q_stop_det: torch.Tensor,   # (T+1,) detached Q(s_i, STOP) predictions, i=0..T
    q_cont_det: torch.Tensor,   # (T+1,) detached Q(s_i, CONT) predictions, i=0..T
    lam: float,
):
    """
    Returns two lists of length T (targets for t = 0, ..., T-1):
        stop_targets[t]  = Qhat_lambda(s_t, STOP)   (eq. 8)
        cont_targets[t]  = Qhat_lambda(s_t, CONT)   (eq. 11 / 12)
    """
    T = len(r_stop) - 1
    stop_targets = []
    cont_targets = []

    for t in range(T):
        stop_targets.append(r_stop[t])

        if t == T - 1:
            cont_targets.append(r_stop[T])
            continue

        horizon = T - t
        q_hats = []
        for n in range(1, horizon + 1):
            if n < horizon:
                known = r_stop[t + 1: t + n]
                candidates = list(known) + [q_stop_det[t + n].item(), q_cont_det[t + n].item()]
            else:
                known = r_stop[t + 1: T]
                r_continue_last = r_stop[T]
                candidates = list(known) + [r_continue_last]
            q_hats.append(max(candidates))

        if horizon == 1:
            cont_target = q_hats[0]
        else:
            weighted_sum = sum((1 - lam) * (lam ** (n - 1)) * q_hats[n - 1] for n in range(1, horizon))
            last_term = (lam ** (horizon - 1)) * q_hats[-1]
            cont_target = weighted_sum + last_term
        cont_targets.append(cont_target)

    return stop_targets, cont_targets


# ============================================================
# 5. LAMBDA SCHEDULE
# ============================================================

def lambda_at_step(step: int, total_steps: int, lam_start: float = 1.0, lam_end: float = 0.1) -> float:
    """Cosine decay of lambda from lam_start to lam_end, matching Table 5 of the paper."""
    if total_steps <= 1:
        return lam_end
    progress = min(step / total_steps, 1.0)
    cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return lam_end + (lam_start - lam_end) * cos_factor


# ============================================================
# 6. TRAINING LOOP
# ============================================================

def train(cfg: Config):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model_name = ENCODER_NAMES[cfg.encoder_variant]

    trajectories = load_trajectories(cfg.data, filter_uninformative=not cfg.no_filter)
    random.shuffle(trajectories)

    n_val = max(1, int(len(trajectories) * cfg.val_frac)) if len(trajectories) > 4 else 0
    val_trajs = trajectories[:n_val]
    train_trajs = trajectories[n_val:]

    print(f"Loaded {len(trajectories)} trajectories "
          f"({len(train_trajs)} train / {len(val_trajs)} val) from {cfg.data}")
    print(f"Encoder backbone: {model_name} (pooling=cls)  state_encoding={cfg.state_encoding}")
    if cfg.state_encoding == "hierarchical":
        avg_docs = sum(len(t.unique_docs) for t in train_trajs) / max(1, len(train_trajs))
        print(f"Avg unique documents per trajectory: {avg_docs:.1f}  "
              f"(evidence_pooling={cfg.evidence_pooling})")

    n_decision_states = sum(len(t.r_stop) - 1 for t in train_trajs)
    n_kept = sum(sum(t.include_mask) for t in train_trajs)
    print(f"Training decision-states: {n_decision_states} total, {n_kept} kept after filtering")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = StopRAGQNetwork(model_name, hidden_size=cfg.head_hidden_size,
                             dropout=cfg.dropout, pooling="cls",
                             state_encoding=cfg.state_encoding,
                             gradient_checkpointing=cfg.gradient_checkpointing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision (fp16): {use_amp}   Gradient checkpointing: {cfg.gradient_checkpointing}   "
          f"batch_size={cfg.batch_size} trajectories")

    train_ds = TrajectoryDataset(train_trajs)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_trajectories
    )

    total_steps = cfg.epochs * max(1, len(train_loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            lam = lambda_at_step(global_step, total_steps, cfg.lambda_start, cfg.lambda_end)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                per_traj_stop, per_traj_cont = run_batch(model, tokenizer, batch, cfg, device)

                loss = 0.0
                n_terms = 0
                for traj, q_stop_i, q_cont_i in zip(batch, per_traj_stop, per_traj_cont):
                    T = len(traj.r_stop) - 1
                    if T == 0:
                        continue

                    # Q(lambda) targets are built from detached, .item()-cast
                    # floats regardless of autocast, so this part is safe
                    # under fp16 (no half-precision accumulation of targets).
                    q_stop_det = q_stop_i.detach().float()
                    q_cont_det = q_cont_i.detach().float()

                    stop_targets, cont_targets = compute_qlambda_targets_for_trajectory(
                        traj.r_stop, q_stop_det, q_cont_det, lam
                    )

                    for t in range(T):
                        if not traj.include_mask[t]:
                            continue
                        target_stop = torch.tensor(stop_targets[t], device=device, dtype=q_stop_i.dtype)
                        target_cont = torch.tensor(cont_targets[t], device=device, dtype=q_cont_i.dtype)

                        loss = loss + (target_stop - q_stop_i[t]) ** 2
                        loss = loss + (target_cont - q_cont_i[t]) ** 2
                        n_terms += 2

                if n_terms == 0:
                    continue

                loss = loss / n_terms

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # so clipping operates on real (unscaled) grad norms
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        val_loss = evaluate(model, tokenizer, val_trajs, device, cfg) if val_trajs else float("nan")
        print(f"epoch {epoch+1}/{cfg.epochs}  lambda={lam:.3f}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "stop_rag_qnetwork.pt"))
    tokenizer.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "encoder_variant": cfg.encoder_variant,
            "state_encoding": cfg.state_encoding,
            "evidence_pooling": cfg.evidence_pooling,
            "head_hidden_size": cfg.head_hidden_size,
            "max_length": cfg.max_length,
        }, f, indent=2)
    print(f"Saved checkpoint to {cfg.output_dir}")


@torch.no_grad()
def evaluate(model, tokenizer, trajectories: List[Trajectory], device, cfg: Config) -> float:
    model.eval()
    total, n = 0.0, 0
    use_amp = cfg.use_amp and device.type == "cuda"
    # small batches to reuse the same run_batch machinery
    for i in range(0, len(trajectories), max(1, cfg.batch_size)):
        batch = trajectories[i:i + max(1, cfg.batch_size)]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            per_traj_stop, per_traj_cont = run_batch(model, tokenizer, batch, cfg, device)
        for traj, q_stop, q_cont in zip(batch, per_traj_stop, per_traj_cont):
            T = len(traj.r_stop) - 1
            if T == 0:
                continue
            stop_targets, cont_targets = compute_qlambda_targets_for_trajectory(
                traj.r_stop, q_stop, q_cont, lam=cfg.lambda_end
            )
            for t in range(T):
                if not traj.include_mask[t]:
                    continue
                total += (stop_targets[t] - q_stop[t].item()) ** 2
                total += (cont_targets[t] - q_cont[t].item()) ** 2
                n += 2
    model.train()
    return total / max(1, n)


# ============================================================
# 7. INFERENCE-TIME CONTROLLER  (drop-in replacement for choose_query_budget)
# ============================================================

class StopRAGController:
    """
    Loads a trained Stop-RAG Q-network and exposes `should_stop` / `q_values`.
    Auto-detects which state encoding the checkpoint was trained with (from
    config.json) and expects the matching argument:

      - state_encoding == "hierarchical" (default/recommended):
            controller.should_stop(question, options, documents=accumulated_documents)
        where `documents` is a list of dicts shaped like your MedRAG
        pipeline's own `accumulated_documents` / `iteration_documents`:
        {"id" or "doc_id": ..., "title": ..., "content": ..., "retrieval_score": ...}

      - state_encoding == "concat" (legacy):
            controller.should_stop(question, options, context=context)
        where `context` is the accumulated evidence string.

    Usage inside your MedRAG loop (hierarchical checkpoint):

        controller = StopRAGController("./stop_rag_ckpt")
        stop = controller.should_stop(question, options, documents=accumulated_documents)
        if stop:
            break  # go straight to final answer generation
        else:
            continue  # run another retrieval iteration
    """

    def __init__(self, checkpoint_dir: str, margin_threshold: float = 0.0, device: Optional[str] = None):
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            cfg = json.load(f)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.state_encoding = cfg.get("state_encoding", "concat")
        self.evidence_pooling = cfg.get("evidence_pooling", "score_weighted")
        self.model = StopRAGQNetwork(cfg["model_name"], hidden_size=cfg["head_hidden_size"],
                                      pooling="cls", state_encoding=self.state_encoding)
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "stop_rag_qnetwork.pt"),
                                                map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.max_length = cfg["max_length"]
        self.margin_threshold = margin_threshold  # tune on a validation set

    @torch.no_grad()
    def q_values(self, question: str, options: Optional[dict],
                 context: Optional[str] = None, documents: Optional[List[dict]] = None):
        if self.state_encoding == "hierarchical":
            if documents is None:
                raise ValueError(
                    "This checkpoint was trained with state_encoding='hierarchical'; "
                    "call q_values(question, options, documents=accumulated_documents) "
                    "with a list of {'id'/'doc_id','title','content','retrieval_score'} dicts, "
                    "not `context=`."
                )
            question_text = build_question_text(question, options)
            pairs = [(question_text, "")]
            for doc in documents:
                pairs.append((doc.get("title", ""), doc.get("content", "")))
            enc = encode_pairs(self.tokenizer, pairs, self.max_length, self.device)
            pooled = self.model.encode(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))

            qvec = pooled[0]
            if len(documents) == 0:
                evid_vec = self.model.no_evidence_embedding
            else:
                doc_embs = pooled[1:]
                scores = [float(d.get("retrieval_score", 0.0)) for d in documents]
                evid_vec = pool_evidence(doc_embs, scores, self.evidence_pooling, self.device)

            combined = torch.cat([qvec, evid_vec], dim=-1).unsqueeze(0)
            q_stop, q_cont = self.model.heads(combined)
            return q_stop.item(), q_cont.item()

        # "concat" (legacy) mode
        if context is None:
            raise ValueError(
                "This checkpoint was trained with state_encoding='concat'; "
                "call q_values(question, options, context=accumulated_evidence_string), "
                "not `documents=`."
            )
        pair = build_state_pair(question, options, context)
        enc = encode_pairs(self.tokenizer, [pair], self.max_length, self.device)
        q_stop, q_cont = self.model.forward_concat(
            enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        return q_stop.item(), q_cont.item()

    def should_stop(self, question: str, options: Optional[dict],
                     context: Optional[str] = None, documents: Optional[List[dict]] = None) -> bool:
        q_stop, q_cont = self.q_values(question, options, context=context, documents=documents)
        # Margin-based decision rule from section 5.3 of the paper:
        # stop iff Q(s, STOP) - Q(s, CONTINUE) exceeds a tuned threshold.
        return (q_stop - q_cont) > self.margin_threshold


# ============================================================
# 8. ENTRY POINT  (no CLI -- edit Config above and run)
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    train(cfg)