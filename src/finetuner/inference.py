from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from finetuner import config


def _setup_logging() -> None:
    """Setup logging configuration for the inference process.

    Configures the root logger with INFO level and a detailed format including
    timestamp, log level, logger name, and message. Only sets up logging if
    no handlers are already configured to avoid duplicate logging.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


logger = logging.getLogger(__name__)


def load_best_model() -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the best trained model and tokenizer from BEST_MODEL_DIR.

    Loads a previously trained LoRA model and its associated tokenizer from the
    specified directory. The function automatically infers the number of labels
    from the saved model configuration.

    Returns:
        A tuple containing:
            - model: Loaded PeftModel with LoRA adapters
            - tokenizer: Associated AutoTokenizer

    Raises:
        FileNotFoundError: If the model directory or required files don't exist.
        ValueError: If the model configuration is invalid.

    Note:
        The function first attempts to read the number of labels from config.json.
        If this fails, it defaults to 5 labels. The base model is loaded from
        MODEL_NAME and then the LoRA adapters are applied from BEST_MODEL_DIR.
    """
    logger.info("Loading best model from: %s", config.BEST_MODEL_DIR)

    if not Path.exists(config.BEST_MODEL_DIR):
        msg = f"Model directory not found: {config.BEST_MODEL_DIR}"
        raise FileNotFoundError(msg)

    tokenizer = AutoTokenizer.from_pretrained(config.BEST_MODEL_DIR)

    config_path = Path(config.BEST_MODEL_DIR) / "config.json"
    if Path.exists(config_path):
        with Path.open(config_path) as f:
            model_config = json.load(f)
        num_labels = model_config.get("num_labels", 5)
    else:
        num_labels = 5

    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
    )

    model = PeftModel.from_pretrained(base_model, config.BEST_MODEL_DIR)

    logger.info("Model loaded successfully with %d labels", num_labels)
    return model, tokenizer


def predict_batch(
    texts: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
) -> list[str]:
    """Run batched prediction on a list of texts.

    Performs inference on a list of input texts using the trained model.
    The function processes texts in batches for memory efficiency and provides
    a progress bar for monitoring.

    Args:
        texts: List of input strings to classify.
        model: Trained PeftModel for inference.
        tokenizer: Tokenizer aligned with the model.
        batch_size: Number of samples per forward pass. Larger batches are
                   more efficient but require more memory.

    Returns:
        List of predicted label strings corresponding to the input texts.

    Note:
        The function automatically detects and uses CUDA if available.
        All predictions are converted from class IDs to label strings using
        the model's id2label configuration.

    Example:
        >>> texts = ["Sample text 1", "Sample text 2"]
        >>> predictions = predict_batch(texts, model, tokenizer, batch_size=8)
        >>> print(predictions)  # ["label1", "label2"]
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("Using device: %s", device)

    predictions: list[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_ids = torch.argmax(logits, dim=1).cpu().tolist()
        cfg = cast("PretrainedConfig", model.config)
        id2label = getattr(cfg, "id2label", None)
        predicted_labels = [
            id2label.get(class_id, str(class_id))
            if isinstance(id2label, dict)
            else str(class_id)
            for class_id in predicted_class_ids
        ]
        predictions.extend(predicted_labels)

    return predictions


def main() -> None:
    """Main function to generate submission.csv from test data.

    Orchestrates the complete inference pipeline for generating predictions
    on test data and creating a submission file. The function performs:

    1. Load the best trained model and tokenizer
    2. Load and validate test data
    3. Generate predictions for all test samples
    4. Create submission DataFrame with ID and label columns
    5. Save submission file to the specified path

    The generated submission file is ready for competition submission and
    contains predictions for all test samples with their corresponding IDs.

    Raises:
        FileNotFoundError: If model directory or test data file doesn't exist.
        ValueError: If test data is missing required columns ('text', 'id').
        RuntimeError: If prediction generation fails.

    Note:
        The function automatically creates the output directory if it doesn't exist.
        Sample predictions are logged for verification purposes.
    """
    _setup_logging()

    logger.info("Starting inference for submission generation")

    model, tokenizer = load_best_model()

    logger.info("Loading test data from: %s", config.TEST_DATA_PATH)
    if not Path.exists(config.TEST_DATA_PATH):
        msg = f"Test data not found: {config.TEST_DATA_PATH}"
        raise FileNotFoundError(msg)

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    logger.info("Loaded %d test samples", len(test_df))

    if "text" not in test_df.columns:
        msg = "Test data must contain 'text' column"
        raise ValueError(msg)
    if "id" not in test_df.columns:
        msg = "Test data must contain 'id' column"
        raise ValueError(msg)

    texts = test_df["text"].astype(str).tolist()

    logger.info("Generating predictions for %d texts", len(texts))
    predictions = predict_batch(texts, model, tokenizer)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "label": predictions,
        },
    )

    logger.info("Saving submission to: %s", config.SUBMISSION_PATH)

    config.SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    logger.info("Submission file saved successfully!")
    logger.info("Submission contains %d predictions", len(submission_df))

    logger.info("Sample predictions:")
    for i in range(min(5, len(submission_df))):
        logger.info(
            "ID: %s -> Label: %s",
            submission_df.iloc[i]["id"],
            submission_df.iloc[i]["label"],
        )


if __name__ == "__main__":
    main()
