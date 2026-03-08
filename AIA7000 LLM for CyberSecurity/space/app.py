import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "jameswatanabegoogle2024/cybersec-qwen"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print("Model ready!")

def ask(question, history):
    prompt = (f"<|im_start|>system\nYou are a cybersecurity expert assistant.<|im_end|>\n"
              f"<|im_start|>user\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=True)
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

gr.ChatInterface(
    fn=ask,
    title="🔐 CyberSec LLM Assistant",
    description="Fine-tuned Qwen2.5-7B on cybersecurity Q&A.",
    examples=[
        "What is a buffer overflow attack?",
        "What does SQL injection exploit?",
        "What is a man-in-the-middle attack?",
    ],
).launch()
