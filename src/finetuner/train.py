from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from finetuner import config, data_utils, model_utils


def _setup_logging() -> None:
    """Setup logging configuration for the training process.

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


def main() -> None:
    """Main training function for XLM-RoBERTa fine-tuning with LoRA.

    Orchestrates the complete training pipeline including data loading, model
    setup, LoRA adaptation, and distributed training. The function performs the
    following steps:

    1. Load and prepare training/development datasets
    2. Initialize tokenizer and tokenize datasets
    3. Create base model and apply LoRA adapters
    4. Configure training arguments for distributed training
    5. Initialize Trainer and run training
    6. Save the best model and tokenizer

    The training uses distributed training with Accelerate, mixed precision
    (FP16) when CUDA is available, and FSDP (Fully Sharded Data Parallel) for
    memory efficiency.

    Raises:
        FileNotFoundError: If training or development data files are not found.
        RuntimeError: If model loading or training fails.

    Note:
        The function automatically detects the number of labels from the
        training data and configures the model accordingly. Training
        checkpoints are saved to OUTPUT_DIR and the best model is saved to
        BEST_MODEL_DIR.
    """
    _setup_logging()

    logger.info(
        "Loading and preparing data from %s and %s",
        config.TRAIN_DATA_PATH,
        config.DEV_DATA_PATH,
    )
    train_dataset, eval_dataset, label2id, id2label = data_utils.load_datasets(
        config.TRAIN_DATA_PATH,
        config.DEV_DATA_PATH,
    )
    num_labels = len(label2id)
    logger.info("Detected %d labels.", num_labels)

    logger.info("Loading tokenizer: %s", config.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    logger.info("Tokenizing datasets (max_length=%d)...", config.MAX_LENGTH)
    tokenized_train, tokenized_eval = data_utils.tokenize_datasets(
        train_dataset,
        eval_dataset,
        tokenizer,
        config.MAX_LENGTH,
    )

    logger.info("Creating base model: %s", config.MODEL_NAME)
    base_model = model_utils.create_model(
        config.MODEL_NAME,
        num_labels,
        id2label,
        label2id,
    )

    logger.info(
        "Applying LoRA adapters (r=%d, alpha=%d, dropout=%.2f)",
        config.LORA_R,
        config.LORA_ALPHA,
        config.LORA_DROPOUT,
    )
    lora_model = model_utils.apply_lora(
        base_model,
        config.LORA_R,
        config.LORA_ALPHA,
        config.LORA_DROPOUT,
    )

    logger.info("Setting up training arguments...")
    use_fsdp = torch.cuda.is_available() and torch.cuda.device_count() > 1
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        fsdp="full_shard auto_wrap" if use_fsdp else "",
    )

    logger.info("Initializing Trainer...")

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=lambda p: model_utils.compute_metrics(
            (p.predictions, p.label_ids),
        ),
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    if trainer.is_world_process_zero():
        logger.info("Saving best model to %s", config.BEST_MODEL_DIR)
        trainer.save_model(str(config.BEST_MODEL_DIR))
        tokenizer.save_pretrained(config.BEST_MODEL_DIR)
        logger.info("Training finished successfully on main process.")


if __name__ == "__main__":
    main()
