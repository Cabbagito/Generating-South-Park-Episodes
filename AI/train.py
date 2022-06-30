from utils import get_trainer, get_tokenizer, get_model,is_gpu_available
from testing import test_model


EPOCHS = 1
BATCH_SIZE = 4
MAX_INPUT_LEN = 512
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.1
FP16 = True


if is_gpu_available():
    print("\nUsing GPU")
else:
    print("\nUsing CPU")

print("\nInitializing Tokenizer")
tokenizer = get_tokenizer()

print("\nInitializing Model")
model = get_model(tokenizer,)


print("\nComputing Metrics Before Training")
# scores_before = test_model(model, tokenizer)

print("\nInitializing Trainer")
trainer = get_trainer(
    model,
    tokenizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    max_input_len=MAX_INPUT_LEN,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler=LR_SCHEDULER_TYPE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
)


trainer.train()

trainer.save_model("model")

print("\nComputing Metrics After Training")
# scores_after = test_model(model, tokenizer)

