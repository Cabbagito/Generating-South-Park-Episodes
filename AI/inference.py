from inference_utils import init

print("***Initializing AI***")
model, tokenizer, pipe = init()
print("***AI Initialized***\n")

print("***Generating***\n")

generated = pipe(
    "|<endoftext>|Scene Description: There is a spear heading towards Kenny with astonishing speed! The spear hits Kenny right in the face and kills him.\nStan: Oh my God, they killed Kenny!\n",
    max_length=1024,
)[0]


print(generated["generated_text"])

