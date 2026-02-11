# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# class MistralLocalClient:
#     def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
#         self.model.to(device)
#         self.device = device

#     def __call__(self, prompt, max_tokens=10, temperature=0.2, stop=None, echo=False):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_tokens,
#                 temperature=temperature,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
#         output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         if echo:
#             return {"choices": [{"text": output_text}]}
#         else:
#             response_text = output_text[len(prompt):].strip()
#             return {"choices": [{"text": response_text}]}
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/Users/shekhar/Desktop/BOT/smartapi-python-main/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

prompt = "RSI: 70, MACD: 0.03, EMA crossover: Yes. What should I do? BUY, SELL or HOLD?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    temperature=0.2,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

print("Model Response:", tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip())
