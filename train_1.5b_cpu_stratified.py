import os
import json
import random
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- 1. Configuration ---
DATA_DIR = Path(r"D:\ai-research-project\fine-tuning\dataset\cyber-security")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "./qwen-1.5b-cyber-v1-stratified-cpu"
TARGET_SAMPLES = 5000

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. Stratified Data Loading (unchanged) ---
print(f"Scanning directory: {DATA_DIR}")
all_files = list(DATA_DIR.rglob("*.json")) + list(DATA_DIR.rglob("*.jsonl"))
samples_by_file = {}

for file_path in all_files:
    file_samples = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if file_path.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        text = ""
                        if "conversations" in row:
                            for msg in row["conversations"]:
                                text += f"{msg.get('from', 'User')}: {msg.get('value', '')}\n"
                        elif "instruction" in row and "output" in row:
                            text = f"<|im_start|>user\n{row['instruction']}<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"
                        elif "text" in row:
                            text = row["text"]
                        if len(text) > 40:
                            file_samples.append({"text": text})
                    except:
                        continue
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "instruction" in item and "output" in item:
                                t = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
                            else:
                                t = item.get("text", "")
                            if len(t) > 40:
                                file_samples.append({"text": t})
    except Exception:
        continue
    if file_samples:
        samples_by_file[file_path.name] = file_samples

num_files = len(samples_by_file)
if num_files == 0:
    raise ValueError("No valid JSON data found. Check your download.")

samples_per_file = max(1, TARGET_SAMPLES // num_files)
stratified_dataset = []
for filename, samples in samples_by_file.items():
    random.shuffle(samples)
    stratified_dataset.extend(samples[:samples_per_file])

random.shuffle(stratified_dataset)
stratified_dataset = stratified_dataset[:TARGET_SAMPLES]

print(f"Successfully prepared {len(stratified_dataset)} examples across {num_files} domains.")
train_dataset = Dataset.from_list(stratified_dataset)

# --- 3. Model Loading ---
print("Loading Model to CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.gradient_checkpointing_enable()

# --- 4. LoRA ---
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# --- 5. Training Arguments (TRL 1.14.2 compatible) ---
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=1,
    use_cpu=True,
    fp16=False,
    bf16=True,
    save_strategy="no",
    max_length=512, 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # 1.14.2 supports processing_class
    args=training_args,
)

print("Starting CPU Training...")
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Success! Adapter saved to {output_dir}")