from inference_utils import init, generate_episode

print("***Initializing AI***")
model, tokenizer = init("model2")
print("***AI Initialized***\n")


prompt = input("Enter Prompt: ")


episode = tokenizer.decode(
    generate_episode(model.cuda(), tokenizer, episode_length=2000, prompt=prompt),
    skip_special_tokens=True,
)

episode = episode.replace("~", "")

# write episode to file
with open("AI/episode.txt", "w") as f:
    f.write(episode)
