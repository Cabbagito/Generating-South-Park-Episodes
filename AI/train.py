from utils import (
    get_trainer,
    get_tokenizer,
    get_model,
    is_gpu_available,
    log,
)


RUNS = 10
EPOCHS = 2
BATCH_SIZE = 4
MAX_INPUT_LEN = 512
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.1
FP16 = True


MODEL_NUM = 8

CHECKPOINT_MODEL = f"model{MODEL_NUM}" if MODEL_NUM is not None else None

if is_gpu_available():
    log("Using GPU")
else:
    log("Using CPU")

log("Initializing Tokenizer")
tokenizer = get_tokenizer()

log("Initializing Model")
model = get_model(tokenizer, checkpoint=CHECKPOINT_MODEL)


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


for run in range(MODEL_NUM + 1, RUNS):
    log(f"Starting Run {run+1}/{RUNS}")
    trainer.train()
    log("Saving Model")
    trainer.save_model(f"AI/model{run}")

