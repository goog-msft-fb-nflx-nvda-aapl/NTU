from datasets import load_dataset, concatenate_datasets
import json
import os

# Load dataset
ds = load_dataset('cais/mmlu', 'computer_security')
all_data = concatenate_datasets([ds['test'], ds['validation'], ds['dev']])

# Convert to instruction format
def format_example(ex):
    choices = ex['choices']
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    answer_letter = chr(65 + ex['answer'])
    answer_text = choices[ex['answer']]
    return {
        "instruction": f"Answer the following cybersecurity question.\n\n{ex['question']}\n\n{choices_text}",
        "output": f"{answer_letter}. {answer_text}"
    }

formatted = [format_example(ex) for ex in all_data]

# Save
out_path = os.path.expanduser("~/cybersec_llm/data/cybersec_qa.json")
with open(out_path, "w") as f:
    json.dump(formatted, f, indent=2)

print(f"Saved {len(formatted)} examples to {out_path}")
print("\nSample:")
print(json.dumps(formatted[0], indent=2))
