from utils import get_dataset
from transformers import pipeline
from datasets import load_metric


def test_model(model, tokenizer):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    test_data = get_dataset(tokenizer, max_len=512, Test=True)
    predictions = []
    references = []

    for data_point in test_data:
        input_ids = data_point["input_ids"]
        length = len(input_ids)
        half_index = int(length / 2)
        model_input = input_ids[:half_index]

        generated_text = pipe(tokenizer.decode(model_input), max_length=length)[0][
            "generated_text"
        ]
        target_text = tokenizer.decode(input_ids)
        predictions.append(generated_text)
        references.append(target_text)

    return compute_rouge(predictions, references)


def compute_rouge(predictions, references):
    rouge = load_metric("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    scores = {
        "rouge1": scores["rouge1"][1],
        "rouge2": scores["rouge2"][1],
        "rougeL": scores["rougeL"][1],
    }
    return scores
