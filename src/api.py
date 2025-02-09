# +
import os
import json
import argparse
import threading
import openai
import pickle
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset

# -
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dnd')
parser.add_argument('--model', type=str,  
                    default='gpt-3.5-turbo-0125')
args = parser.parse_args()


def read_key():
    with open("assets/openai.key") as f:
        return f.read()


# +
def dump_pkl(content, filename):
    with open(filename, "wb") as file:
        pickle.dump(content, file)

def load_pkl(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# -
def load_personas(path="../data/personas.json"):
    with open(path) as file:
        return json.load(file)



def chat_completion_with_retry(message, seconds=10):
    max_retries = 10
    result = None
    for retry in range(max_retries):
        def request_completion():
            nonlocal result
            result = openai.chat.completions.create(
                model=args.model, 
                messages=message,
                max_tokens=300,
                stop=["\n"],
        )
        thread = threading.Thread(target=request_completion)
        thread.start()
        thread.join(timeout=seconds)  # Wait for 10 seconds
        if not thread.is_alive():
            break  # Command completed successfully
        else:
            print(f"Attempt {retry + 1} took more than "+str(seconds)+" seconds. Retrying...")
    return result



def format_instruction(question, persona):
    if persona is None:
        message = [
            {"role": "user", "content": question}
        ]
        return message
    
    if args.model in ["mixtral"]:
        message = [
            {
                "role": "user", 
                "content": f"You are a helpful assistant. I am {persona}.\n\n{question}"
            }
        ]
    else:
        message = [
            {
                "role": "system", 
                "content": f"You are a helpful assistant. I am {persona}."
            },
            {
                "role": "user", 
                "content": question
            }
        ]
    return message


def load_personas(path="../data/personas.json"):
    with open(path) as file:
        return json.load(file)


def load_dna(persona):
    dataset = []
    df = pd.read_csv("../data/dnd/do_not_answer_en.csv")
    for idx, row in df.iterrows():
        question = row['question']
        instruction = format_instruction(question, persona)
        dataset.append((instruction, row['types_of_harm']))
    return dataset


# strong reject
def load_SR(persona):
    dataset = []
    df = pd.read_csv("../data/strongreject/strongreject_dataset.csv")
    for idx, row in df.iterrows():
        question = row['forbidden_prompt']
        instruction = format_instruction(question, persona)
        dataset.append((instruction, row['category']))
    return dataset


# GSM-8K
def format_question(template, row):
    formatted_text = template.format(question=row['question']).strip()
    return formatted_text

def load_gsm8k(persona):
    # Load the dataset
    dataset = load_dataset("gsm8k", 'main')
    gsm_df = pd.DataFrame(dataset["test"])
    # Split the answer and explanation if needed
    gsm_df[["explain", "answer"]] = gsm_df["answer"].str.split("####", expand=True)
    question_template = """Answer the given question and show your work first.\
    You must output only the answer in your final sentence like ‘‘Therefore, the answer is ...’’.\
    Question is {question}"""

    # Format questions according to the template
    formatted_questions = []
    for idx, row in gsm_df.iterrows():
        question = format_question(question_template, row)
        message =  format_instruction(question, persona)
        formatted_questions.append((message, row['answer']))
    return formatted_questions

# MMLU
mmlu_path = "../data/mmlu/dev/"
def load_mmlu(persona, path=mmlu_path):
    question_template= """Answer the given multiple choice question and show your work first.
        The answer can only be an option like (A), (B), (C), (D).
        You need to output the answer in your final sentence like ‘‘Therefore, the answer is ...’’
        Question is  {question}
        (A) {option1} (B) {option2} (C) {option3} (D) {option4}""" 

    files = os.listdir(path)
    formatted_questions = []
    for file in (files):
        mmlu_df = pd.read_csv(os.path.join(path, file), 
                              names=['Question', 'A', 'B', 'C', 'D', 'Answer'])
        for idx, row in mmlu_df.iterrows():
            question = format_question(question_template, row)
            message =  format_instruction(question, persona)
            formatted_questions.append((message, row['Answer']))
    return formatted_questions

dataset = load_dnd(None)
all_personas = load_personas()


def get_responses(dataset):
    responses = []
    for qs, _ in tqdm(dataset):
        response = chat_completion_with_retry(qs)
        responses.append(response)
    return responses


def process(dataset_name, model_name):
    # no persona
    if dataset_name == 'dnd':
        dataset = load_dnd(None)
    elif dataset_name == 'SR':
        dataset = load_SR(None)
    elif dataset_name == 'gsm8k':
        dataset = load_gsm8k(None)
    elif dataset_name == 'mmlu':
        dataset = load_mmlu(None)

    # dataset = load_dnd(None)
    dataset = load_SR(None)
    save_path = os.path.join(out_dir, f"{dataset_name}_{model_name}_no_persona.pkl")
    print(f"save_path: {save_path}")
    responses = get_responses(dataset)
    dump_pkl(responses, save_path)
    
    # get personas
    for category, personas in all_personas.items():
        for p in personas:        
            save_path = os.path.join(
                out_dir, 
                f"{dataset_name}_{model_name}_{category.lower()}_{p.lower()}.pkl"
            )
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                continue

            if dataset_name == 'dna':
                dataset = load_dna(p)
            elif dataset_name == 'SR':
                dataset = load_SR(p)
            elif dataset_name == 'gsm8k':
                dataset = load_gsm8k(p)
            elif dataset_name == 'mmlu':
                dataset = load_mmlu(p)
            else:
                raise Exception("Dataset not recognized!")

            responses = get_responses(dataset)
            dump_pkl(responses, save_path)


if __name__ == '__main__':
    openai.api_key = read_key()
    out_dir = f"results/{args.dataset}/{args.model}/"
    os.makedirs(out_dir, exist_ok=True)
    process(args.dataset, args.model)
