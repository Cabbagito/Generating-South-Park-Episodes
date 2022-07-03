from inference_utils import init, generate_episode

print("***Initializing AI***")
model, tokenizer = init("model2")
print("***AI Initialized***\n")


prompt = input("Enter Prompt: ")


episode = tokenizer.decode(
    generate_episode(model.cuda(), tokenizer, episode_length=4000, prompt=prompt,max_model_batch=1024),
    skip_special_tokens=True,
)

episode = episode.replace("~", "")

with open("GeneratedEpisodes/episode.txt", "w") as f:
    f.write(episode)
