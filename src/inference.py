from __future__ import annotations

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

import config


def predict_batch(
    texts: list[str],
    model: PeftModel | AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
) -> list[str]:
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
        predicted_labels = [
            model.config.id2label[class_id] for class_id in predicted_class_ids
        ]
        predictions.extend(predicted_labels)

    return predictions


if __name__ == "__main__":
    print("Loading model for inference...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=5,
    )

    model = PeftModel.from_pretrained(base_model, config.BEST_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.BEST_MODEL_DIR)

    sample_texts = [
        "Это первый текст для проверки работы классификатора.",
        "Какой прекрасный день, чтобы дообучать модели!",
        "Третий пример, чтобы убедиться, что батчинг работает.",
        "Краткое сообщение.",
        "А это очень длинное предложение, которое, возможно, будет обрезано токенизатором, потому что его длина может превысить максимальное значение, установленное в конфигурации, но мы это предусмотрели.",
    ]

    print(f"Found {len(sample_texts)} texts to classify.")

    predictions = predict_batch(sample_texts, model, tokenizer)

    for text, pred in zip(sample_texts, predictions, strict=False):
        print(f"\nТекст: '{text}'")
        print(f"  -> Предсказанная метка: {pred}")
