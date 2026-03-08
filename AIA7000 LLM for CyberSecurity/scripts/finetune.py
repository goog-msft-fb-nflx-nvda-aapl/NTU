import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Load data
with open("/home/jtan/cybersec_llm/data/cybersec_qa.json") as f:
    data = json.load(f)

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# Format into chat template
def format_prompt(ex):
    return {
        "text": f"<|im_start|>system\nYou are a cybersecurity expert assistant.<|im_end|>\n"
                f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
    }

dataset = dataset.map(format_prompt)
print("Dataset ready. Sample:")
print(dataset[0]['text'])
print("\nStarting training...")

# Load tokenizer & model
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Training config
training_args = SFTConfig(
    output_dir="/home/jtan/cybersec_llm/outputs",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    dataset_text_field="text",
    
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
trainer.save_model("/home/jtan/cybersec_llm/models/cybersec-qwen")
print("Training complete! Model saved.")
