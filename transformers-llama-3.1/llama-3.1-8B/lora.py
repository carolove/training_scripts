from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
from utils.prompter import Prompter

data = load_dataset("json", data_files="/workspace/training_scripts/transformers-llama-3.1/llama-3.1-8B/dataset/huanhuan.json")
prompter = Prompter("alpaca")

tokenizer = AutoTokenizer.from_pretrained('/workspace/checkpoints/Meta-Llama-3___1-8B-Instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 256
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    # could be sped up, probably
    return tokenized_full_prompt


train_val = data["train"].train_test_split(
    test_size=100, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)

model = AutoModelForCausalLM.from_pretrained('/workspace/checkpoints/Meta-Llama-3___1-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
model.print_trainable_parameters()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="/workspace/checkpoints/llama3_1_instruct_lora",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=10, # 为了快速演示，这里设置10，建议你设置成100
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    max_steps=20
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()