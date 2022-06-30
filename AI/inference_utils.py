from utils import get_tokenizer,get_model
from transformers import pipeline

def init():
    tokenizer = get_tokenizer()
    model = get_model(tokenizer,use_local=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, pipe