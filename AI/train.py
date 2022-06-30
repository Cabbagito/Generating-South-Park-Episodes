from utils import get_trainer, get_tokenizer, get_model, is_gpu_available, log
from testing import test_model


RUNS = 10
EPOCHS = 2
BATCH_SIZE = 4
MAX_INPUT_LEN = 512
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.1
FP16 = True


if is_gpu_available():
    log("Using GPU")
else:
    log("Using CPU")

log("Initializing Tokenizer")
tokenizer = get_tokenizer()

log("Initializing Model")
model = get_model(tokenizer)


log("Computing Metrics Before Training")
scores_before = test_model(model, tokenizer)
print(scores_before)

log("Initializing Trainer")
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

scores_after = []

for run in range(RUNS):
    log(f"Starting Run {run+1}/{RUNS}")
    trainer.train()
    trainer.save_model(f"AI/model{run}")

    log("Computing Metrics After Training")
    scores_after.append(test_model(model, tokenizer))
    print(scores_after[len(scores_after) - 1])

