from utils import get_tokenizer, get_model
from tqdm import tqdm


def init(model_name=None):
    tokenizer = get_tokenizer()
    model = get_model(tokenizer, checkpoint=model_name)
    return model, tokenizer


def generate_episode(
    model, tokenizer, episode_length=9000, prompt="", max_model_batch=512
):

    episode = tokenizer.encode(prompt)
    for i in tqdm(range(episode_length // max_model_batch + 1)):

        if i == 0:
            if prompt != "":
                episode = [tokenizer.bos_token_id] + episode
            else:
                episode.append(tokenizer.bos_token_id)

        if i == episode_length // max_model_batch:
            episode.append(tokenizer.encode("~")[0])

        model_input = episode[-(max_model_batch // 2) :] if i != 0 else episode
        model_input = tokenizer(tokenizer.decode(model_input), return_tensors="pt")

        model_input = {k: model_input[k].to("cuda") for k in model_input.keys()}

        generated_text = model.generate(
            **model_input,
            max_length=max_model_batch,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            do_sample=True,
        )[0]
        generated_text = generated_text.cpu().numpy().tolist()
        episode += (
            generated_text[-int(max_model_batch / 2) :]
            if i != 0
            else generated_text[len(episode) :]
        )
    return episode

