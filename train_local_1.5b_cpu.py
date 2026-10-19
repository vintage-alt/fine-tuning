import os
import glob
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from itertools import islice

max_seq_length = 1024 # Keep it small to save RAM

print("Loading model on CPU...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load in 16-bit (CPU doesn't support bitsandbytes 4-bit easily)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32, # CPU needs full precision
    device_map="cpu"
)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load Local Data
print("Loading local data...")
BASE_DATA_DIR = "D:/New Volume/cyber-security"
all_files = glob.glob(os.path.join(BASE_DATA_DIR, "**", "*.json*"), recursive=True)
dataset = load_dataset("json", data_files=all_files, split="train", streaming=True)

def format_prompt(example):
    instruction = example.get("Instruction", example.get("instruction", ""))
    output = example.get("Output", example.get("output", ""))
    if not output and "conversations" in example:
        convs = example["conversations"]
        if isinstance(convs, list):
            instruction = next((c.get("value", "") for c in convs if c.get("from") == "human"), "")
            output = next((c.get("value", "") for c in convs if c.get("from") == "gpt"), "")
    if instruction and output:
        return {"text": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"}
    return {"text": ""}

def generate_valid_data(ds):
    for example in ds:
        formatted = format_prompt(example)
        if len(formatted["text"]) > 0:
            yield formatted

# Take only 1000 rows for CPU training to save time
dataset = list(islice(generate_valid_data(dataset), 1000))
dataset = Dataset.from_list(dataset)
print(f"Loaded {len(dataset)} rows.")

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=20, # Very small for CPU
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        use_cpu=True
    ),
)

print("Starting CPU training (this will be slow)...")
trainer.train()

adapter_name = "qwen2.5-1.5b-cyber-lora-cpu"
model.save_pretrained(adapter_name)
tokenizer.save_pretrained(adapter_name)
print("Done!")