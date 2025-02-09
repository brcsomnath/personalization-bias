# +
import os
import json
import pickle
import argparse
import pandas as pd

from tqdm import tqdm
from transformers import pipeline

# -

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    type=str,
                   default='llama-70b')
parser.add_argument('--dataset',
                    type=str,
                   default='dna')
parser.add_argument('--device_id', 
                    type=int,
                    default=0)
parser.add_argument('--run_id', 
                    type=int,
                    default=0)
args = parser.parse_args()


def dump_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate():
    out_dir = os.path.join("results", args.dataset, args.model)
    classifier = pipeline(model='LibrAI/longformer-action-ro', 
                          device=args.device_id,
                          batch_size=300)
    
    results = {}
    for filename in tqdm(os.listdir(out_dir)):
        if filename.endswith(f"{args.run_id}.csv"):
            df = pd.read_csv(os.path.join(out_dir, filename))
            preds = [str(p) for p in df['Prediction']]
            scores = classifier(preds)
            results[filename] = [s['label'] for s in scores]
            
        if filename.endswith(".pkl"):
            df = load_pkl(os.path.join(out_dir, filename))
            preds = [p.choices[0].message.content for p in df]
            scores = classifier(preds)
            results[filename] = [s['label'] for s in scores]
    
    file_out = os.path.join(out_dir, f"results_{args.run_id}.json")
    dump_json(results, file_out)
    return results


if __name__ == '__main__':
    evaluate()


