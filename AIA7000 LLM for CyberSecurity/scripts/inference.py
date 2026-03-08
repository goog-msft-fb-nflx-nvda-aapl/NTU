import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "/home/jtan/cybersec_llm/models/cybersec-qwen"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

def ask(question):
    prompt = (f"<|im_start|>system\nYou are a cybersecurity expert assistant.<|im_end|>\n"
              f"<|im_start|>user\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Q: {question}\nA: {response}\n{'-'*50}")

ask("What is a buffer overflow attack?")
ask("Which attack involves sending malformed packets to crash a system?")
ask("What does SQL injection exploit?")
