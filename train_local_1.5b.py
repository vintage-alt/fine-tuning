import os
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. Settings ---
# Use forward slashes to avoid Windows path-escaping bugs
DATA_DIR = Path(r"D:/ai-research-project/fine-tuning/dataset/cyber-security")

# Choose "Qwen/Qwen2.5-7B-Instruct" for best capability, or "Qwen/Qwen2.5-1.5B-Instruct" for faster training
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
OUTPUT_DIR = "./cyber_qwen_lora_adapter"
MAX_SEQ_LENGTH = 1024
MAX_SAMPLES = 5000  # Adjust as needed

# --- 2. Robust Local Data Loading ---
print(f"Scanning directory: {DATA_DIR}")
all_json_files = list(DATA_DIR.rglob("*.json")) + list(DATA_DIR.rglob("*.jsonl"))
print(f"Found {len(all_json_files)} JSON/JSONL files.")

valid_samples = []

for file_path in all_json_files:
    if len(valid_samples) >= MAX_SAMPLES:
        break
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if file_path.suffix == ".jsonl":
                for line in f:
                    if len(valid_samples) >= MAX_SAMPLES:
                        break
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
                            valid_samples.append({"text": text})
                    except json.JSONDecodeError:
                        continue
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if len(valid_samples) >= MAX_SAMPLES:
                            break
                        if isinstance(item, dict):
                            if "instruction" in item and "output" in item:
                                t = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
                            else:
                                t = item.get("text", "")
                            if len(t) > 40:
                                valid_samples.append({"text": t})
    except Exception:
        continue

print(f"Prepared {len(valid_samples)} training examples.")
if not valid_samples:
    raise ValueError(f"No valid JSON records found in {DATA_DIR}. Check directory contents.")

dataset = Dataset.from_list(valid_samples)

# --- 3. 4-bit Quantization (QLoRA) Setup for 16GB VRAM ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# --- 4. LoRA Configuration ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    max_steps=300,
    fp16=False,
    bf16=True,  # Modern 16GB GPUs support bfloat16
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
)

# --- 6. Execute & Save ---
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuning complete. Adapter saved to: {OUTPUT_DIR}")