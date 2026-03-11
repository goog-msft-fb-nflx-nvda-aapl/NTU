from typing import List
import re
    
def get_inference_system_prompt() -> str:
    return "You are a helpful assistant. Answer the question based on the provided passages. If the answer cannot be found, reply CANNOTANSWER."

def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    context = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context_list)])
    return f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

def parse_generated_answer(pred_ans: str) -> str:
    if "Answer:" in pred_ans:
        return pred_ans.split("Answer:")[-1].strip()
    return pred_ans.strip()