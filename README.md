# SecOps-AI: Fine-Tuning Qwen 2.5 (1.5B) for Cybersecurity

## 📌 Project Overview
This project focuses on fine-tuning a Large Language Model (LLM) to act as a specialized Cybersecurity/SecOps assistant. The goal is to enable the model to parse, understand, and generate complex cybersecurity JSON schemas, specifically focusing on penetration testing methodologies, scope areas, and testing types.

## 🛠️ Tech Stack
*   **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
*   **Dataset:** `Vyber07/cyber-security` (Hugging Face Hub)
*   **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) via PEFT
*   **Libraries:** Hugging Face `transformers`, `datasets`, `peft`, `trl`
*   **Environment:** Google Colab (NVIDIA T4 GPU) for training; Local Windows environment for inference testing.

## ⚙️ Training Configuration
To ensure the model could handle the complex nested JSON data without running out of memory, the following configurations were used:
*   **LoRA Settings:** `r=16`, `lora_alpha=32`
*   **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (Expanded from the standard q/v for deeper schema comprehension)
*   **Batch Size:** 4 (with 2 gradient accumulation steps)
*   **Epochs:** 1 (5,000 highly stratified samples)
*   **Max Sequence Length:** 512 tokens
*   **Optimization:** `fp16` (Half-precision for T4 GPU acceleration)

## 🚧 Engineering Challenges Overcome
1.  **CPU Bottleneck & Triton Kernel Crash:** Initial attempts to train locally on Windows CPU failed due to TRL's new `fused_lm_head` Triton kernel throwing `RuntimeError: 0 active drivers`. This was bypassed by moving the training pipeline to a cloud GPU and downgrading `trl` to a stable version (`1.14.2`).
2.  **Environment Corruption:** Google Colab's pre-installed `pandas` and `cudf` conflicted with newly installed NLP libraries, resulting in a `_pandas_parser_CAPI` circular import error. This was resolved by isolating dependencies and upgrading `torchao` to match `peft` requirements.
3.  **Nested JSON Data Format:** The dataset did not have standard `instruction/output` columns. It was deeply nested with columns like `methodologies` and `scope_areas`. The data pipeline was re-engineered to serialize the entire row using `json.dumps()` to teach the LLM how to read raw security JSON.

## 🚀 Usage
The fine-tuned LoRA adapter is saved locally. To use the model, load the base Qwen 2.5 model and attach the adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "./qwen-1.5b-cyber-v1-stratified-gpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
