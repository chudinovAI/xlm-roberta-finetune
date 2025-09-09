from __future__ import annotations

import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification


def create_model(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str] | None,
    label2id: dict[str, int] | None,
) -> AutoModelForSequenceClassification:
    """
    Create a base sequence classification model.

    Args:
        model_name: Hugging Face model identifier.
        num_labels: Number of target labels.
        id2label: Mapping from numeric id to label string.
        label2id: Mapping from label string to numeric id.

    Returns:
        An instantiated AutoModelForSequenceClassification.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def apply_lora(
    model: AutoModelForSequenceClassification,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> PeftModel:
    """
    Wrap a base model with LoRA adapters.

    Args:
        model: Base HF classification model.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        A PeftModel with LoRA adapters applied.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    peft_model: PeftModel = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """
    Compute evaluation metrics for Trainer.

    Args:
        eval_pred: Tuple of (logits, labels) as numpy arrays.

    Returns:
        Dictionary with accuracy and weighted F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted")),
    }
