import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json



class StopRAGQNetwork(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        hidden_size=256,
        dropout=0.2,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        enc_dim = self.encoder.config.hidden_size

        
        self.stop_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.cont_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # [CLS] representation
        pooled = outputs.last_hidden_state[:, 0, :]

        stop_q = self.stop_head(pooled).squeeze(-1)
        cont_q = self.cont_head(pooled).squeeze(-1)

        return stop_q, cont_q

class TrainedStopRAGController:
    """
    Inference wrapper around the trained Stop-RAG Q-network.
    """

    def __init__(
        self,
        checkpoint_path="./stop_rag_ckpt/best_stop_rag_qnetwork.pt",
        model_name="microsoft/deberta-v3-base",
        max_length=512,
        device=None
    ):

        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.max_length = max_length

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(
            f"[Stop-RAG] Loading tokenizer: "
            f"{model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        print(
            f"[Stop-RAG] Loading Q-network..."
        )

        self.model = StopRAGQNetwork(
            model_name=model_name
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        config_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "config.json"
        )

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.margin_threshold = float(
                config.get("margin_threshold", 0.0)
            )
        else:
            self.margin_threshold = 0.0

        print(
            f"[Stop-RAG] Margin threshold: "
            f"{self.margin_threshold}"
        )

        

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = (
                    checkpoint["model_state_dict"]
                )

            elif "state_dict" in checkpoint:

                state_dict = (
                    checkpoint["state_dict"]
                )

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"[Stop-RAG] Loaded successfully."
        )

        print(
            f"[Stop-RAG] Device: "
            f"{self.device}"
        )

    
    def build_state(
        self,
        question,
        options,
        evidence
    ):
        

        
        parts = [f"Question: {question.strip()}"]

        if options:
            opt_str = " ".join(
                f"({key}) {options[key]}"
                for key in sorted(options.keys())
            )
            parts.append(f"Options: {opt_str}")

        question_text = " ".join(parts)

        
        evidence = (evidence or "").strip()

        if evidence:
            evidence_text = evidence
        else:
            evidence_text = "(no evidence retrieved yet)"

        return question_text, evidence_text

    

    @torch.no_grad()
    def predict_q_values(
        self,
        question,
        options,
        evidence
    ):

        question_text, evidence_text = (
            self.build_state(
                question,
                options,
                evidence
            )
        )

        encoded = self.tokenizer(
            question_text,
            evidence_text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value in encoded.items()
        }

        q_stop, q_continue = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids", None),
        )

        return {
            "q_stop":
                float(
                    q_stop.item()
                ),

            "q_continue":
                float(
                    q_continue.item()
                )
        }

    
    @torch.no_grad()
    def decide(
        self,
        question,
        options,
        evidence
    ):
        q_values = self.predict_q_values(
            question=question,
            options=options,
            evidence=evidence
        )

        q_stop = q_values["q_stop"]
        q_continue = q_values["q_continue"]

        margin = q_stop - q_continue

        if margin > self.margin_threshold:
            action = "STOP"
        else:
            action = "CONTINUE"

        return {
            "action": action,
            "q_stop": q_stop,
            "q_continue": q_continue,
            "margin": margin,
            "margin_threshold": self.margin_threshold
        }