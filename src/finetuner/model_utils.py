from __future__ import annotations

from typing import cast

import numpy as np
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, PreTrainedModel


def create_model(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str] | None,
    label2id: dict[str, int] | None,
) -> AutoModelForSequenceClassification:
    """Create a base sequence classification model.
    Initializes a Hugging Face AutoModelForSequenceClassification with the specified
    configuration for text classification tasks.

    Args:
        model_name: Hugging Face model identifier (e.g., "FacebookAI/xlm-roberta-large").
        num_labels: Number of target classification labels.
        id2label: Optional mapping from numeric ID to label string. If None,
                 default labels will be generated.
        label2id: Optional mapping from label string to numeric ID. If None,
                 default mappings will be generated.

    Returns:
        An instantiated AutoModelForSequenceClassification model ready for training.

    Example:
        >>> model = create_model(
        ...     "FacebookAI/xlm-roberta-large",
        ...     num_labels=5,
        ...     id2label={0: "class1", 1: "class2"},
        ...     label2id={"class1": 0, "class2": 1}
        ... )
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
) -> PeftModel | PeftMixedModel:
    """Apply LoRA (Low-Rank Adaptation) adapters to a base model.

    Wraps a base classification model with LoRA adapters for parameter-efficient
    fine-tuning. This significantly reduces the number of trainable parameters
    while maintaining model performance.

    Args:
        model: Base Hugging Face classification model to adapt.
        r: LoRA rank (dimensionality of the low-rank matrices). Higher values
           increase model capacity but also trainable parameters.
        lora_alpha: LoRA alpha scaling factor. Controls the magnitude of
                   LoRA updates relative to the base model.
        lora_dropout: Dropout probability for LoRA layers to prevent overfitting.

    Returns:
        A PeftModel with LoRA adapters applied and trainable parameters printed.

    Note:
        The function automatically prints the number of trainable parameters
        to help monitor the efficiency of the LoRA adaptation.

    Example:
        >>> base_model = create_model("bert-base-uncased", num_labels=3)
        >>> lora_model = apply_lora(base_model, r=16, lora_alpha=32, lora_dropout=0.1)
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    base: PreTrainedModel = cast("PreTrainedModel", model)
    peft_model: PeftModel | PeftMixedModel = get_peft_model(base, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def compute_metrics(
    eval_pred: tuple[
        np.ndarray | tuple[np.ndarray, ...],
        np.ndarray | tuple[np.ndarray, ...],
    ],
) -> dict[str, float]:
    """Compute evaluation metrics for the Trainer.

    Calculates accuracy and weighted F1 score for model evaluation during training.
    This function is designed to be used as a compute_metrics callback in the
    Hugging Face Trainer.

    Args:
        eval_pred: Tuple containing (logits, labels) as numpy arrays.
                  - logits: Model predictions with shape (batch_size, num_labels)
                  - labels: Ground truth labels with shape (batch_size,)

    Returns:
        Dictionary containing:
            - "accuracy": Classification accuracy (0.0 to 1.0)
            - "f1_weighted": Weighted F1 score (0.0 to 1.0)

    Note:
        The weighted F1 score accounts for class imbalance by weighting each
        class's F1 score by its support (number of true instances).

    Example:
        >>> metrics = compute_metrics((logits, labels))
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"F1 Score: {metrics['f1_weighted']:.3f}")
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    predictions = np.argmax(logits, axis=-1)
    y_true = np.array(labels).ravel()
    y_pred = np.array(predictions).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
