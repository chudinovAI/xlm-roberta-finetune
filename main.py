from __future__ import annotations

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

import config
import data_utils
import model_utils


def main() -> None:
    train_dataset, eval_dataset, label2id, id2label = data_utils.load_and_prepare_data(
        config.DATA_PATH,
    )
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenized_train, tokenized_eval = data_utils.tokenize_datasets(
        train_dataset,
        eval_dataset,
        tokenizer,
        config.MAX_LENGTH,
    )

    base_model = model_utils.create_model(
        config.MODEL_NAME,
        num_labels,
        id2label,
        label2id,
    )
    lora_model = model_utils.apply_lora(
        base_model,
        config.LORA_R,
        config.LORA_ALPHA,
        config.LORA_DROPOUT,
    )

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        fsdp="full_shard auto_wrap",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=model_utils.compute_metrics,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        print(f"Saving the best model to {config.BEST_MODEL_DIR}")
        trainer.save_model(config.BEST_MODEL_DIR)
        tokenizer.save_pretrained(config.BEST_MODEL_DIR)
        print("Training finished successfully on the main process!")


if __name__ == "__main__":
    main()
