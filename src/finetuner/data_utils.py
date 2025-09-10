from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


def load_datasets(
    train_path: str | Path,
    dev_path: str | Path,
) -> tuple[Dataset, Dataset, dict[str, int], dict[int, str]]:
    """Load and prepare training and development datasets.

    Loads CSV files containing training and development data, creates label mappings
    based on training data, and converts the data to Hugging Face Dataset format.

    Args:
        train_path: Path to the training CSV file containing 'text' and 'label' columns.
        dev_path: Path to the development CSV file containing 'text' and 'label' columns.

    Returns:
        A tuple containing:
            - train_dataset: Training dataset in Hugging Face Dataset format
            - dev_dataset: Development dataset in Hugging Face Dataset format
            - label2id: Dictionary mapping label strings to integer IDs
            - id2label: Dictionary mapping integer IDs to label strings

    Raises:
        FileNotFoundError: If either train_path or dev_path files don't exist.
        KeyError: If required columns ('text', 'label') are missing from the data.
    """
    try:
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
    except FileNotFoundError as e:
        logging.exception(f"Failed to load CSV files: {e}")
        raise

    labels = train_df["label"].astype(str).unique().tolist()
    label2id: dict[str, int] = {label: i for i, label in enumerate(labels)}
    id2label: dict[int, str] = dict(enumerate(labels))

    train_df["label"] = train_df["label"].astype(str).replace(label2id)
    dev_df["label"] = dev_df["label"].astype(str).replace(label2id)

    return (
        Dataset.from_pandas(train_df),
        Dataset.from_pandas(dev_df),
        label2id,
        id2label,
    )


def tokenize_datasets(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> tuple[Dataset, Dataset]:
    """Tokenize training and evaluation datasets.

    Applies tokenization to text data in both training and evaluation datasets
    using the provided tokenizer with specified maximum length.

    Args:
        train_dataset: Training dataset containing 'text' column.
        eval_dataset: Evaluation dataset containing 'text' column.
        tokenizer: Hugging Face tokenizer for text processing.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A tuple containing:
            - tokenized_train: Tokenized training dataset
            - tokenized_eval: Tokenized evaluation dataset

    Note:
        The tokenization uses 'max_length' padding and truncation to ensure
        consistent sequence lengths across all samples.
    """

    def tokenize_function(examples: dict[str, list]) -> BatchEncoding:
        """Tokenize a batch of examples.

        Args:
            examples: Dictionary containing 'text' key with list of strings.

        Returns:
            BatchEncoding object with tokenized inputs.
        """
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    return tokenized_train, tokenized_eval
