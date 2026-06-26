# Spring 2026, 535510 Reinforcement Learning
# HW4: DPO
# Instructor: Ping-Chun Hsieh (National Yang Ming Chiao Tung University)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use GPU 0; change to "1", "2", ... as needed

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Inspection
# ══════════════════════════════════════════════════════════════════════════════

dataset       = load_dataset("trl-lib/ultrafeedback_binarized")
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

# TODO — Print and inspect 
# ########## Your Code (2-3 lines)##########



# ########## End of Your Code ###########
# Expected keys: "prompt", "chosen", "rejected"
# Each of "chosen" and "rejected" is a list of {"role": ..., "content": ...} dicts.

# Average token lengths (first 1000 examples)
_tokenizer_for_stats = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def assistant_text(turns):
    """Extract the assistant turn content from a conversation list."""
    for turn in turns:
        if turn["role"] == "assistant":
            return turn["content"]
    return ""

chosen_lengths   = []
rejected_lengths = []
for ex in train_dataset.select(range(1000)):
    chosen_lengths.append(
        len(_tokenizer_for_stats(assistant_text(ex["chosen"]))["input_ids"])
    )
    rejected_lengths.append(
        len(_tokenizer_for_stats(assistant_text(ex["rejected"]))["input_ids"])
    )

avg_chosen   = sum(chosen_lengths)   / len(chosen_lengths)
avg_rejected = sum(rejected_lengths) / len(rejected_lengths)
print(f"Avg chosen   token length: {avg_chosen:.1f}")
print(f"Avg rejected token length: {avg_rejected:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# Initialize Weight & Bias
# ══════════════════════════════════════════════════════════════════════════════

wandb.init(
    project = "dpo",
    name    = "default",
    config  = {
        "model":                       "Qwen/Qwen2.5-0.5B-Instruct",
        "beta":                        0.1,
        "learning_rate":               5e-7,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs":            1,
        "max_length":                  1024,
        # data stats logged for reference
        "avg_chosen_token_length":     round(avg_chosen,   1),
        "avg_rejected_token_length":   round(avg_rejected, 1),
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# Model & Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Load model and tokenizer
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ══════════════════════════════════════════════════════════════════════════════
# Training Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Fill in baseline hyperparameters
training_args = DPOConfig(
    output_dir                  = "dpo-default",
    num_train_epochs            = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,    # effective batch = 8
    learning_rate               = 5e-7,
    beta                        = 0.1,
    max_length                  = 1024,
    logging_steps               = 25,
    eval_strategy               = "steps",
    eval_steps                  = 100,
    save_strategy               = "epoch",
    bf16                        = True,
    report_to                   = "wandb",  # hands all trainer metrics to W&B
    run_name                    = "default",
)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

# TODO: Instantiate DPOTrainer
# Check DPOTrainer and provide a proper initialization
# Such as model, args, processing_class, train_dataset, eval_dataset, and so on
# Note that ref_model is left unset since DPOTrainer shall create a frozen copy automatically
# ########## Your Code (5-10 lines)##########





# ########## End of Your Code ###########

# Train
trainer.train()

# Peak VRAM (log to W&B as a summary metric)
peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM used: {peak_vram_gb:.2f} GB")
wandb.summary["peak_vram_gb"] = peak_vram_gb

# Save checkpoints
trainer.save_model("dpo-baseline-final")
tokenizer.save_pretrained("dpo-baseline-final")
print("Model saved to dpo-baseline-final/")

wandb.finish()