import hashlib
import json
import os
import shutil
import sys

import torch
from datasets import Dataset
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
                          Trainer, TrainingArguments)

# -----------------------
# Config
# -----------------------
MAX_LENGTH  = 128
MODEL_DIR   = "tinyLLM"
DATA_FILE   = "train.txt"

# Persisted inside MODEL_DIR
STATE_FILE    = os.path.join(MODEL_DIR, "training_state.json")
SNAPSHOT_FILE = os.path.join(MODEL_DIR, "data_snapshot.txt")

epochs = 200

# -----------------------
# Tokenizer
# -----------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# -----------------------
# State helpers
# -----------------------
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(full_data_hash: str, mode: str, active_hash: str, epochs_done: int) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "full_data_hash": full_data_hash,
                "training_mode": mode,           # "full" | "append"
                "active_data_hash": active_hash, # hash of the portion actually trained
                "epochs_completed": epochs_done,
            },
            f,
            indent=2,
        )


def load_snapshot() -> str | None:
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_snapshot(content: str) -> None:
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def clear_checkpoints() -> None:
    if not os.path.isdir(MODEL_DIR):
        return
    removed = []
    for entry in os.listdir(MODEL_DIR):
        if entry.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(MODEL_DIR, entry), ignore_errors=True)
            removed.append(entry)
    if removed:
        print(f"  Cleared checkpoints: {', '.join(removed)}")


def find_latest_checkpoint() -> str | None:
    if not os.path.isdir(MODEL_DIR):
        return None
    checkpoints = [d for d in os.listdir(MODEL_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(MODEL_DIR, checkpoints[-1])


# -----------------------
# Dataset builder
# -----------------------
def build_example(prompt: str, completion: str) -> dict:
    full_text  = f"{prompt}\n{completion}{tokenizer.eos_token}"
    prompt_text = f"{prompt}\n"

    full_tokens   = tokenizer(full_text,   truncation=True, max_length=MAX_LENGTH)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)

    input_ids      = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    prompt_len     = len(prompt_tokens["input_ids"])

    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def load_examples_from_content(content: str) -> list[dict]:
    blocks   = [b.strip() for b in content.split("\n\n")]
    examples = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        examples.append(build_example(lines[0], lines[1]))
    return examples


# -----------------------
# Custom Collator
# -----------------------
class CustomCollator:
    def __init__(self, tok):
        self.tokenizer = tok

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch   = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = [l + [-100] * (max_len - len(l)) for l in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ===========================================================
# MAIN LOGIC
# ===========================================================
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Read the current training file
with open(DATA_FILE, "r", encoding="utf-8") as f:
    current_content = f.read()

current_full_hash = compute_hash(current_content)
state             = load_state()

# -----------------------
# Decision tree
# -----------------------
examples_to_train: list[dict] = []
training_mode:     str        = "full"
active_data_hash:  str        = current_full_hash
checkpoint_path:   str | None = None

print("=" * 55)

# ── Case 1: very first run (no state file yet) ──────────────
if not state:
    print("[FIRST RUN] No prior state found. Training on all data.")
    examples_to_train = load_examples_from_content(current_content)
    training_mode     = "full"
    active_data_hash  = current_full_hash

# ── Case 2: same file ───────────────────────────────────────
elif current_full_hash == state.get("full_data_hash"):
    epochs_done = state.get("epochs_completed", 0)

    if epochs <= epochs_done:
        print(
            f"[SKIP] Same data, already trained {epochs_done} epoch(s) "
            f"(requested {epochs}). Nothing to do."
        )
        sys.exit(0)

    else:
        remaining = epochs - epochs_done
        print(
            f"[RESUME] Same data — extending training: "
            f"{epochs_done} → {epochs} epochs (+{remaining} more)."
        )
        examples_to_train = load_examples_from_content(current_content)
        training_mode     = state["training_mode"]
        active_data_hash  = state["active_data_hash"]
        checkpoint_path   = find_latest_checkpoint()
        if checkpoint_path:
            print(f"  Resuming from checkpoint: {checkpoint_path}")

# ── Case 3: file has changed ────────────────────────────────
else:
    old_snapshot = load_snapshot()

    # ── 3a: pure append (old content is an exact prefix) ────
    is_append = (
        old_snapshot is not None
        and len(current_content) > len(old_snapshot)
        and current_content.startswith(old_snapshot)
    )

    if is_append:
        new_portion      = current_content[len(old_snapshot):]
        new_portion_hash = compute_hash(new_portion)
        new_examples     = load_examples_from_content(new_portion)

        if not new_examples:
            print(
                "[SKIP] New content was appended but produced no parseable "
                "training examples (possibly just whitespace). Nothing to do."
            )
            sys.exit(0)

        # Was this exact append already trained?
        same_append_already_trained = (
            state.get("training_mode") == "append"
            and state.get("active_data_hash") == new_portion_hash
        )

        if same_append_already_trained:
            epochs_done = state.get("epochs_completed", 0)
            if epochs <= epochs_done:
                print(
                    f"[SKIP] Same appended data already trained {epochs_done} "
                    f"epoch(s) (requested {epochs}). Nothing to do."
                )
                sys.exit(0)
            else:
                remaining = epochs - epochs_done
                print(
                    f"[RESUME-APPEND] Same appended data — extending: "
                    f"{epochs_done} → {epochs} epochs (+{remaining} more)."
                )
                checkpoint_path = find_latest_checkpoint()
                if checkpoint_path:
                    print(f"  Resuming from checkpoint: {checkpoint_path}")
        else:
            print(
                f"[APPEND] {len(new_examples)} new example(s) detected. "
                "Training on ALL data (old + new) to prevent forgetting. "
                "Checkpoints reset."
            )
            clear_checkpoints()

        # ── CRITICAL: always train on the FULL dataset (old + new) ──────────
        # Training only on new examples causes catastrophic forgetting — the
        # model's weights are overwritten and it loses everything it learned
        # before. By including all examples in every epoch, each gradient step
        # reinforces both old and new knowledge simultaneously.
        examples_to_train = load_examples_from_content(current_content)
        training_mode     = "append"
        active_data_hash  = new_portion_hash   # still tracks the new-data hash
                                                # for change-detection next run

    # ── 3b: data modified or completely replaced ─────────────
    else:
        print(
            "[RETRAIN] Training data changed (modified or replaced). "
            "Training on all data from scratch (checkpoints reset)."
        )
        clear_checkpoints()
        examples_to_train = load_examples_from_content(current_content)
        training_mode     = "full"
        active_data_hash  = current_full_hash

print("=" * 55)

if not examples_to_train:
    print("[WARN] Example list is empty — nothing to train on. Exiting.")
    sys.exit(1)

dataset      = Dataset.from_list(examples_to_train)
dataset_size = len(dataset)
print(f"Examples to train: {dataset_size}")

# -----------------------
# Training scaling
# -----------------------
steps_per_epoch = max(1, dataset_size // 2)
logging_steps   = max(1, steps_per_epoch // 5)
save_steps      = steps_per_epoch
warmup_steps    = max(10, int(0.1 * steps_per_epoch * epochs))

print(f"Epochs: {epochs}  |  Steps/epoch: {steps_per_epoch}")
print("=" * 55)

# -----------------------
# Model (load or create)
# -----------------------
config_path = os.path.join(MODEL_DIR, "config.json")

if os.path.exists(config_path):
    print("Loading existing model weights...")
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
    num_train_epochs=epochs,        # Trainer handles remaining epochs on resume
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
# Trainer + run
# -----------------------
data_collator = CustomCollator(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=checkpoint_path)

# -----------------------
# Save model + state
# -----------------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

save_state(
    full_data_hash=current_full_hash,
    mode=training_mode,
    active_hash=active_data_hash,
    epochs_done=epochs,
)
save_snapshot(current_content)  # always snapshot the full file for future append detection

print("=" * 55)
print(f"[DONE] Model saved to '{MODEL_DIR}'. State updated.")
print(f"  mode={training_mode}  epochs_completed={epochs}")
print("=" * 55)
