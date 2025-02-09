# +
import os
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    type=str,
                   default='llama2-70B')
args = parser.parse_args()

# -

def load_json(path):
    with open(path) as f:
        return json.load(f)
    

def pb(util, safe):
    points = []
    for k in util.keys():
        if k == 'no_persona' or k.lower() not in safe.keys():
            continue
        points.append([100-100*np.mean(safe[k.lower()]), np.mean(util[k])])
    
    points = np.array(points)
    no_persona = np.mean(points, axis=0) #np.array(no_persona)
    diff = np.sqrt(np.sum(np.mean((points - no_persona)**2, axis=0)))
    return (diff)

def pb_util(util):
    points = []
    for k in util.keys():
        points.append([np.mean(util[k])])
    
    points = np.array(points)
    no_persona = np.mean(points, axis=0) #np.array(no_persona)
    diff = np.sqrt(np.sum(np.mean((points - no_persona)**2, axis=0)))
    return (diff)


def pb_safe(safe):
    points = []
    for k in safe.keys():
        points.append([100-100*np.mean(safe[k.lower()])])
    
    points = np.array(points)
    mean = np.mean(points, axis=0)
    diff = np.sqrt(np.sum(np.mean((points - mean)**2, axis=0)))
    return diff


if __name__ == '__main__':

    utility = load_json(f"results/mmlu/{args.model}/results.json")
    safety = load_json(f"results/dna/{args.model}/scores.json")



    print(f"PB Score: {pb(utility, safety)}")
    print(f"PB Score (Utility): {pb_util(utility)}")
    print(f"PB Score (Safety): {pb_safe(safety)}")
