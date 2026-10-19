import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Configuration
model_name = "Qwen/Qwen2.5-7B-Instruct"
dataset_name = "Vyber07/cyber-security"
output_dir = "./qwen-7b-cyber-v1-cpu"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Extreme Memory-Saving Model Load
# We use bfloat16 to halve the memory footprint from 28GB to 14GB.
print("Loading Model to CPU (This will take a lot of RAM)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, 
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Gradient checkpointing saves RAM by trading it for computation time
model.gradient_checkpointing_enable()

# 3. Apply LoRA (Low-Rank Adaptation)
# r=4 is exceptionally low, specifically chosen to prevent CPU RAM crashes
peft_config = LoraConfig(
    r=4, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 4. Load Dataset via Streaming
# Streaming prevents the 16GB dataset from loading into RAM
print("Connecting to Vyber07 dataset...")
dataset = load_dataset(dataset_name, split="train", streaming=True)

# Pull exactly 5,000 samples for the V1 test
subset = dataset.take(5000)

def format_prompt(example):
    # Adjust this formatting based on the specific JSON structure of the subset you pull
    text = f"<|im_start|>user\n{example.get('instruction', 'Analyze this security scenario:')}<|im_end|>\n<|im_start|>assistant\n{example.get('output', example.get('text', ''))}<|im_end|>"
    return {"text": text}

# Convert the streaming dataset to a fixed list so SFTTrainer can calculate epochs
print("Downloading and formatting 5,000 samples...")
formatted_data = [format_prompt(row) for row in subset]
from datasets import Dataset
train_dataset = Dataset.from_list(formatted_data)

# 5. CPU Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, # MUST be 1 for CPU
    gradient_accumulation_steps=8, # Simulates a batch size of 8
    max_steps=100, # Start with just 100 steps to see if it survives
    learning_rate=2e-4,
    logging_steps=1,
    use_cpu=True,
    fp16=False,
    bf16=True, # Modern CPUs process bfloat16 efficiently
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512, # Kept strictly short to save RAM
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting CPU Training...")
trainer.train()

# 6. Save the Adapter
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Success! V1 Adapter saved to {output_dir}")