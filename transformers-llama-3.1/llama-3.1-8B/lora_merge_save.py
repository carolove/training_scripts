from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/workspace/checkpoints/Meta-Llama-3___1-8B-Instruct'
lora_path = '/workspace/checkpoints/llama3_1_instruct_lora/checkpoint-20' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 合并lora权重
model = model.merge_and_unload()

# 保存合并后的模型
model.save_pretrained('/workspace/checkpoints/llama3_1_instruct_lora_merged_model')