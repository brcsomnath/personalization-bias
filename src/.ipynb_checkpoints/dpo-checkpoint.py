# +
import os
import json
import torch
import random
import argparse

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments
)


from tqdm import tqdm
from copy import deepcopy
from trl import DPOTrainer
from huggingface_hub import login
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model




# ### Dataset
parser = argparse.ArgumentParser()
parser.add_argument('--config', 
                    type=str,
                    default="dpo",
                    help='Training config.')
parser.add_argument('--model', 
                    type=str,
                    default="llama",
                    help='Model name.')
args = parser.parse_args()



def get_hf_token(filename="../assests/hf.key"):
    with open(filename) as f:
        return f.read()


def load_json(path):
    with open(path) as f:
        raw_data = json.load(f)
        personas = []
        for k, v in raw_data.items():
            personas.extend(v)
    personas.append("")
    return personas



def dpo_sys(example):
    example['system'] = ""
    return example


def assign_random_persona(example):
    persona = random.choice(personas)
    if len(persona) > 0:
        example['system'] = f"You are a helpful assistant. I am {persona}."
    else:
        example['system'] = ""
    return example


def add_all_persona(train_ds):
    aug_ds = []
    for ex in tqdm(train_ds):
        sample_set = random.sample(personas, 3)
        for persona in sample_set:
            add_ex = deepcopy(ex)
            add_ex['system'] = f"You are a helpful assistant. I am {persona}."
            aug_ds.append(add_ex)
    aug_ds = Dataset.from_list(aug_ds)
    ds = concatenate_datasets([train_ds, aug_ds])
    return ds


def chatml_format(example):
    # format system
    if len(example['system'])>0:
        message ={
            "role":"system",
            "content":example['system']
        }
        system=tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system=""
    
    
    #format instruction
    message={"role": "user", "content":example['question']}
    prompt=tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    
    # format chosen answer
    chosen =example['chosen']+"<im_end>\n"
    
    # format rejected answer
    rejected = example['rejected']+"<im_end>\n"
    
    return {
        "prompt": system+prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


if __name__ == '__main__':
    # Login to Huggingface
    login(token=get_hf_token())

    personas = load_json("../data/personas.json")
    output_dir = f"../outputs/{args.model}-{args.config}/"

    # ### Model
    if args.model == 'mistral':
        model_name="teknium/OpenHermes-2.5-Mistral-7B"
    elif args.model == 'llama':
        model_name = "meta-llama/Llama-2-7b-chat-hf"

    train_ds=load_dataset("Intel/orca_dpo_pairs")["train"]

    if args.config == 'dpo-random':
        train_ds = train_ds.map(assign_random_persona)
    elif args.config == 'dpo-vanilla':
        train_ds = train_ds.map(dpo_sys)
    elif args.config == 'dpo':
        pass # train_ds remains unchanged


    original_columns=train_ds.column_names

    tokenizer=AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side="left"

    train_dataset=train_ds.map(
        function=chatml_format,
        remove_columns=original_columns
    )

    # ### Training

    peft_config=LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'k_proj',
            'gate_proj',
            'v_proj',
            'up_proj',
            'q_proj',
            'o_proj',
            'down_proj'
        ]
    )

    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model.config.use_cache=False
    model=get_peft_model(model, peft_config)
    model.get_memory_footprint()



    training_args=TrainingArguments(
        gradient_checkpointing=True,
        gradient_accumulation_steps=5,
        remove_unused_columns=False,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_steps=100,
        save_strategy="steps",
        save_steps=25,
        logging_steps=5,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        bf16=False, 
        fp16=True,
        warmup_steps=50,
    )

    dpo_trainer=DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )

    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(
        os.path.join(output_dir, "post_dpo_vanilla_ckpt")
    )
