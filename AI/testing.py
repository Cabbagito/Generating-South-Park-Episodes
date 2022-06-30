from utils import get_dataset, getcwd


def test_model(model, tokenizer):
    data = get_dataset(tokenizer, max_len=512, Test=True)
    pass

