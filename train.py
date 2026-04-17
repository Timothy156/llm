import os
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# -----------------------
# Config
# -----------------------
MAX_LENGTH = 128
MODEL_DIR = "tinyLLM"

# -----------------------
# Tokenizer
# -----------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# Dataset builder
# -----------------------
def build_example(prompt, completion):
    full_text = f"{prompt}\n{completion}{tokenizer.eos_token}"
    prompt_text = f"{prompt}\n"

    full_tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)
    prompt_len = len(prompt_tokens["input_ids"])

    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_examples_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n")]
    examples = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        examples.append(build_example(lines[0], lines[1]))

    return examples


examples = load_examples_from_file("train.txt")
dataset = Dataset.from_list(examples)

dataset_size = len(dataset)
print(f"Dataset size: {dataset_size}")

# -----------------------
# Smart training scaling
# -----------------------
if dataset_size < 50:
    epochs = 100
elif dataset_size < 200:
    epochs = 50
elif dataset_size < 1000:
    epochs = 20
else:
    epochs = 5

steps_per_epoch = max(1, dataset_size // 2)

logging_steps = max(1, steps_per_epoch // 5)
save_steps = steps_per_epoch

warmup_steps = max(10, int(0.1 * steps_per_epoch * epochs))

print(f"Epochs: {epochs}")
print(f"Steps/epoch: {steps_per_epoch}")

# -----------------------
# Custom Collator
# -----------------------
class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt"
        )

        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for l in labels:
            padded = l + [-100] * (max_len - len(l))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


data_collator = CustomCollator(tokenizer)

# -----------------------
# Model (AUTO LOAD OR CREATE)
# -----------------------
if os.path.exists(MODEL_DIR):
    print("Loading existing model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
else:
    print("Creating new model...")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_ctx=256,
        n_embd=384,
        n_layer=6,
        n_head=6,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

# -----------------------
# Training args
# -----------------------
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=epochs,
    learning_rate=5e-4,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=2,
    fp16=False,
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,
)

# -----------------------
# Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# -----------------------
# Resume training if possible
# -----------------------
checkpoint_path = None

if os.path.isdir(MODEL_DIR):
    checkpoints = [d for d in os.listdir(MODEL_DIR) if d.startswith("checkpoint")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        checkpoint_path = os.path.join(MODEL_DIR, checkpoints[-1])
        print(f"Resuming from checkpoint: {checkpoint_path}")

trainer.train(resume_from_checkpoint=checkpoint_path)

# -----------------------
# Save
# -----------------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)