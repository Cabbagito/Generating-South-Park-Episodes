from utils import get_dataset
from datasets import load_metric
from tqdm import tqdm


def test_model(model, tokenizer, max_len=512):
    test_data = get_dataset(tokenizer, max_len=max_len, Test=True)
    predictions = []
    references = []
    model_inputs = []
    counter = 0
    for data_point in tqdm(test_data):
        counter += 1
        if counter == 51:
            break
        input_ids = data_point["input_ids"]
        length = len(input_ids)
        half_index = int(length / 2)

        model_input = tokenizer.decode(input_ids[:half_index])

        model_inputs.append(model_input)

        target_text = tokenizer.decode(input_ids)

        references.append(target_text)

    BATCH_SIZE = 50
    for i in tqdm(range(0, len(model_inputs), BATCH_SIZE)):
        model_input = tokenizer(
            model_inputs[i : i + BATCH_SIZE], padding=True, return_tensors="pt"
        )
        model_input = {k: model_input[k].to("cuda") for k in model_input.keys()}
        generated_text = model.generate(
            **model_input, max_length=max_len, pad_token_id=tokenizer.eos_token_id
        )
        for t in range(generated_text.shape[0]):
            predictions.append(tokenizer.decode(generated_text[t]))
    return compute_rouge(predictions, references)


def compute_rouge(predictions, references):
    rouge = load_metric("rouge")
    scores = rouge.compute(
        predictions=[pred[len(predictions) // 2 :] for pred in predictions],
        references=[ref[len(references) // 2 :] for ref in references],
    )
    scores = {
        "rouge1": scores["rouge1"][1],
        "rouge2": scores["rouge2"][1],
        "rougeL": scores["rougeL"][1],
    }
    return scores

