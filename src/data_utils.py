from __future__ import annotations

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase


def load_and_prepare_data(
    file_path: str,
) -> tuple[Dataset, Dataset, dict[str, int], dict[int, str]]:
    df = pd.read_csv(file_path)
    labels = df["label"].astype(str).unique().tolist()
    label2id: dict[str, int] = {label: i for i, label in enumerate(labels)}
    id2label: dict[int, str] = dict(enumerate(labels))
    df["label"] = df["label"].astype(str).replace(label2id)
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    return (
        Dataset.from_pandas(train_df),
        Dataset.from_pandas(eval_df),
        label2id,
        id2label,
    )


def tokenize_datasets(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> tuple[Dataset, Dataset]:
    def tokenize_function(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    return tokenized_train, tokenized_eval
