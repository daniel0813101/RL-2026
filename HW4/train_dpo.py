# Spring 2026, 535510 Reinforcement Learning
# HW4: DPO
# Instructor: Ping-Chun Hsieh (National Yang Ming Chiao Tung University)

import os
import argparse
from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-0.5B-Instruct with DPO.")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_name", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--project", default="dpo")
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--output_dir", default="dpo-default")
    parser.add_argument("--save_dir", default="dpo-baseline-final")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Inspection
# ══════════════════════════════════════════════════════════════════════════════

dataset       = load_dataset(args.dataset_name)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

# TODO — Print and inspect 
# ########## Your Code (2-3 lines)##########
print("Dataset keys:", list(train_dataset[0].keys()))
print("First training example:")
pprint(train_dataset[0])


# ########## End of Your Code ###########
# Expected keys: "prompt", "chosen", "rejected"
# Each of "chosen" and "rejected" is a list of {"role": ..., "content": ...} dicts.

# Average token lengths (first 1000 examples)
_tokenizer_for_stats = AutoTokenizer.from_pretrained(args.model_name)

def assistant_text(turns):
    """Extract the assistant turn content from a conversation list."""
    for turn in turns:
        if turn["role"] == "assistant":
            return turn["content"]
    return ""

chosen_lengths   = []
rejected_lengths = []
num_stats_examples = min(1000, len(train_dataset))
for ex in train_dataset.select(range(num_stats_examples)):
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
    project = args.project,
    name    = args.run_name,
    config  = {
        "model":                       args.model_name,
        "beta":                        args.beta,
        "learning_rate":               args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs":            args.num_train_epochs,
        "max_length":                  args.max_length,
        "max_steps":                   args.max_steps,
        # data stats logged for reference
        "avg_chosen_token_length":     round(avg_chosen,   1),
        "avg_rejected_token_length":   round(avg_rejected, 1),
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# Model & Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = args.model_name

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
    output_dir                  = args.output_dir,
    num_train_epochs            = args.num_train_epochs,
    per_device_train_batch_size = args.per_device_train_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation_steps,
    learning_rate               = args.learning_rate,
    beta                        = args.beta,
    max_length                  = args.max_length,
    logging_steps               = args.logging_steps,
    eval_strategy               = "steps",
    eval_steps                  = args.eval_steps,
    save_strategy               = "epoch",
    bf16                        = args.bf16,
    report_to                   = "wandb",  # hands all trainer metrics to W&B
    run_name                    = args.run_name,
    max_steps                   = args.max_steps,
)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

# TODO: Instantiate DPOTrainer
# Check DPOTrainer and provide a proper initialization
# Such as model, args, processing_class, train_dataset, eval_dataset, and so on
# Note that ref_model is left unset since DPOTrainer shall create a frozen copy automatically
# ########## Your Code (5-10 lines)##########
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)




# ########## End of Your Code ###########

# Train
trainer.train()

# Peak VRAM (log to W&B as a summary metric)
peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
print(f"Peak VRAM used: {peak_vram_gb:.2f} GB")
wandb.summary["peak_vram_gb"] = peak_vram_gb

# Save checkpoints
trainer.save_model(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"Model saved to {args.save_dir}/")

wandb.finish()
