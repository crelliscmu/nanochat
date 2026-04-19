import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = os.path.expandvars("$HOME/nano_789/nanochat_artifacts/d20_drope_50/hf_sft")
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_path)

#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to("cuda")
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
conversation = [
        {"role": "user", "content": "How far is the moon from the earth?"},
    ]

inputs = tokenizer.apply_chat_template( conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
print(output)