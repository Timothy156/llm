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

# -----------------------
# Tokenizer
# -----------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-2 has no pad token → reuse EOS
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# Dataset builder
# -----------------------
def build_example(prompt, completion):
    full_text = f"{prompt}\n{completion}{tokenizer.eos_token}"
    prompt_text = f"{prompt}\n"

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Get prompt length
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prompt_len = len(prompt_tokens["input_ids"])

    # Build labels (mask prompt)
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
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if len(lines) < 2:
            continue

        prompt = lines[0]
        completion = lines[1]

        examples.append(build_example(prompt, completion))

    return examples


examples = load_examples_from_file("train.txt")
dataset = Dataset.from_list(examples)

# -----------------------
# Custom Collator (CRITICAL FIX)
# -----------------------
class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f["labels"] for f in features]

        # Remove labels for tokenizer padding
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        # Pad inputs
        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt"
        )

        max_len = batch["input_ids"].shape[1]

        # Pad labels manually
        padded_labels = []
        for l in labels:
            padded = l + [-100] * (max_len - len(l))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


data_collator = CustomCollator(tokenizer)

# -----------------------
# Model
# -----------------------
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
# Training
# -----------------------
training_args = TrainingArguments(
    output_dir="tinyLLM",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=50,
    learning_rate=5e-4,
    warmup_steps=50,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    fp16=False,  # CPU safe
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,  # remove warning
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# -----------------------
# Save
# -----------------------
model.save_pretrained("tinyLLM")
tokenizer.save_pretrained("tinyLLM")