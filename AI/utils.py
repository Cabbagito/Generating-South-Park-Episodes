from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.cuda import is_available
from pickle import load
from random import shuffle


def get_trainer(
    model,
    tokenizer,
    train_dataset=None,
    epochs=1,
    batch_size=2,
    max_input_len=512,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    lr_scheduler="cosine",
    weight_decay=0.1,
    fp16=True,
):

    args = TrainingArguments(
        output_dir="AI/checkpoints",
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=get_data_collator(tokenizer),
        tokenizer=tokenizer,
        train_dataset=train_dataset
        if train_dataset is not None
        else get_dataset(tokenizer, max_len=max_input_len),
    )

    return trainer


def is_gpu_available():
    return is_available()


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = [tokenizer.eos_token_id]
    return tokenizer


def get_model(tokenizer, checkpoint=None):
    if checkpoint is not None:
        name = f"AI/{checkpoint}"
    else:
        name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(
        name, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
    )
    model.pad_token_id = tokenizer.eos_token_id
    return model


def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt"
    )


def get_dataset(tokenizer, max_len=512, Test=False):

    log("Preparing Data")
    file = "Data/SouthPark_Data_test.pkl" if Test else "Data/SouthPark_Data_train.pkl"
    data = load(open(file, "rb"))

    data_dict = {"input_ids": []}
    for k, v in data.items():
        input_ids = tokenizer.encode(v[2])
        input_ids.insert(0, tokenizer.bos_token_id)
        input_ids.append(tokenizer.eos_token_id)

        for i in range(0, len(input_ids), max_len):
            if i + max_len + 1 >= len(input_ids):
                data_dict["input_ids"].append(tokenizer.encode("~") + input_ids[i:])
            else:
                data_dict["input_ids"].append(input_ids[i : i + max_len])
    shuffle(data_dict["input_ids"])
    return Dataset.from_dict(data_dict)


def log(text):
    print(50 * "*")
    print(text)
    print(50 * "*")
    print()

