# +
import io
import os
import re
import json
import pickle
import argparse
import itertools
import contextlib
import multiprocessing

import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# -
parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    type=str,
                   default='llama2-70B')
parser.add_argument('--dataset',
                    type=str,
                   default='mmlu')
args = parser.parse_args()


def extract_answer_mmlu(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return 'NA'
    
    patterns = [
        r"the\s*answer\s*is\s*\(([A-D])\)", 
        r"the\s*answer[,.\s]*is\s*([A-D])", 
        r"the\s*correct\s*answer\s*is\s*\(([A-D])\)",  
        r"the\s*answer\s*is:\s*\n*\s*\(([A-D])\)",  
        r"ption\s\(([A-D])\)",  
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return 'NA'


# GSM8K
def extract_answer_gsm8k(text):
    """
    Extracts the answer from a string that follows the phrase "Therefore, the answer is..."
    or finds the last number in the string if the phrase is not found.
    Converts the extracted number to a float.
    If no number is found, returns "NA".
    """
    # Remove commas for thousands separators
    text = text.replace(',', '')
    
    # First, try to find a sentence with "Therefore, the answer is..." and extract the number after it
    answer_pattern = r"Therefore, the answer is.*?(\d+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    
    if match:
        answer_str = match.group(1)
    else:
        # If not found, find the last number in the entire string
        all_numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if all_numbers:
            answer_str = all_numbers[-1]
        else:
            return "NA"
    
    try:
        # Convert the extracted number to a float
        answer = float(answer_str)
        return answer
    except ValueError:
        # If conversion to float fails, return "NA"
        return "NA"


# MBPP
def extract_python_code(text):
    answers =  re.findall(r"```(?:python\n)?([\s\S]*?)\d*```",text)
    if len(answers) > 0:
        return answers[0]
    else:
        return ""


def preprocess_code(code):
    """Pre-processes the code by commenting out lines that contain `input(...)`."""
    processed_lines = []
    for line in code.split('\n'):  # Split the code into lines
        # Check if 'input(' is in the line, and comment it out if so
        if 'input(' in line:
            processed_lines.append('# ' + line)
        else:
            processed_lines.append(line)

    # Reassemble the code
    processed_code = '\n'.join(processed_lines)
    return processed_code


def run_code_and_tests(code, test_cases, results):
    """
    Function to be executed in a separate process.
    Evaluates the code snippet and corresponding assert statements.
    """
    code = preprocess_code(code)
    try:
        with io.StringIO() as dummy_output:
            with contextlib.redirect_stdout(dummy_output):
                exec(code)
                for test in test_cases:
                    try:
                        exec(test)
                        results.put(1)
                    except AssertionError:
                        results.put(0)
                    except Exception:
                        results.put(0)
    except:
        results.put("NA")


def evaluate_code_and_tests(code, test_cases, timeout_seconds=1):
    """
    Evaluates a code snippet and its corresponding assert statements with a timeout.
    Suppresses print statements from the executed code.
    Returns a list of [1, 0, NA] based on the outcomes or ["NA", "NA", "NA"] if timed out.
    """

    results_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=run_code_and_tests, args=(code, test_cases, results_queue))
    proc.start()
    
    proc.join(timeout_seconds)
    
    
    if proc.is_alive():
        proc.terminate() 
        proc.join()
        return ["NA", "NA", "NA"]
    
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    while len(results) < len(test_cases):
        results.append("NA")
    
    return results



def format_question(template, row):
    formatted_text = template.format(
        question=row['Question'],
        option1=row['A'],
        option2=row['B'],
        option3=row['C'],
        option4=row['D']
    )
    return formatted_text


def load_mmlu(path='data/mmlu/dev/', persona='a physically disabled person',sys_template=""""""):
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
            formatted_question = format_question(question_template, row)
            message = [
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant. I am {persona}."
                },
                {
                    "role": "user", 
                    "content":  formatted_question,
                }
            ]
            formatted_questions.append((message, row['Answer']))
    return formatted_questions

# load GSM8K
def load_gsm8k(persona):
    dataset = load_dataset("gsm8k", 'main')
    gsm_df = pd.DataFrame(dataset["test"])

    # Split the answer and explanation if needed
    gsm_df[["explain", "answer"]] = gsm_df["answer"].str.split("####", expand=True)
    question_template = """Answer the given question and show your work first.
            You must output only the answer in your final sentence like ‘‘Therefore, the answer is ...’’.
            Question is {question}
    """


    # Format questions according to the template
    formatted_questions = []
    for idx, row in gsm_df.iterrows():
        formatted_question = format_question(question_template, row)
        message = [
                {
                    "role": "user", 
                    "content": f"You are a helpful assistant. I am {persona}.\n\n{formatted_question}"
                }
            ]
        formatted_questions.append((message, row['answer']))
    return formatted_questions

# load MBPP
def load_mbpp(persona):
    dataset = load_dataset("mbpp")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mbpp_df = pd.concat([pd.DataFrame(dataset[i]) for i in dataset])
    question_template = """Write a python program for the following problem:
                        {question}

                        Your code should pass these tests:
                        {tests}
    """
    
    # Format questions according to the template 
    formatted_questions = []
    for _, row in mbpp_df.iterrows():
        formatted_question = format_question(question_template, row)
        message = [
                {
                    "role": "user", 
                    "content": f"You are a helpful assistant. I am {persona}.\n\n{formatted_question}",
                }
            ]
        message = tokenizer.apply_chat_template(message, tokenize=False)
        formatted_questions.append((message, row['test_list']))
        
    return formatted_questions

# +
def load_personas(path="../data/personas.json"):
    with open(path) as file:
        persona_dict = json.load(file)
        
        all_personas = []
        for k, v in persona_dict.items():
            all_personas.extend(v)
        return all_personas 


if __name__ == '__main__':
    all_personas = load_personas()


    if args.dataset == 'mmlu':
        dataset = load_mmlu()
        extract_answer = extract_answer_mmlu
    elif args.dataset == 'gsm8k':
        dataset = load_gsm8k()
        extract_answer = extract_answer_gsm8k
    elif args.dataset == 'mbpp':
        dataset = load_mbpp()
        extract_answer = extract_python_code
    else:
        raise Exception("Dataset not recognized!")


    num_runs = 3

    user_means = []
    user_means_list = {}

    for persona in all_personas:
        print(f"[{persona}]")
        
        accuracies = []
        for iteration in list(range(num_runs)):
            directory = f"../results/{args.dataset}/{args.model}"
            output_file = os.path.join(directory, f'{args.dataset}_{args.model}_{persona}_{iteration}.txt')  

            if os.path.exists(output_file):
                with open(output_file, 'rb') as file:
                    all_outputs = pickle.load(file)
                if not args.model.startswith("gpt"):
                    all_outputs = list(itertools.chain.from_iterable(all_outputs))
                
                predictions = [extract_answer(output) for output in all_outputs]
                labels = [ex[1] for ex in dataset]

                if args.dataset == 'mbpp':
                    ground_truths = [[1, 1, 1] for _ in predictions]
                    predicted_codes=[]
                    for code, tests in tqdm(zip(predictions, labels)):
                        predicted_codes.append(evaluate_code_and_tests(code, tests))
                    predicted_codes = [item for sublist in predicted_codes for item in sublist]
                    ground_truths = [item for sublist in ground_truths for item in sublist]
                    accuracy = np.mean([pred == label for pred, label in zip(predictions, ground_truths)])
                else:
                    accuracy = np.mean([pred == label for pred, label in zip(predictions, labels)])
                accuracies.append(accuracy * 100)


        # Calculate mean accuracy
        mean_accuracy = np.mean(accuracies) if accuracies else 0
        user_means.append(mean_accuracy)
        user_means_list[persona] = accuracies


    filename = f'../results/{args.dataset}/{args.model}/results.json'

    # Save results
    with open(filename, 'w') as f:
        json.dump(user_means_list, f)


