# Exploring Safety-Utility Trade-Offs in Personalized Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-green``.svg)](https://opensource.org/licenses/MIT)


We present the implementation of the NAACL 2025 paper:

> [**Exploring Safety-Utility Trade-Offs in Personalized Language Models**](https://arxiv.org/pdf/2406.11107),<br/>
[Anvesh Rao Vijjini*](https://nvshrao.github.io/), [Somnath Basu Roy Chowdhury*](https://www.cs.unc.edu/~somnath/), and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/). <br>
UNC Chapel Hill

## Overview
As large language models (LLMs) become increasingly integrated into daily applications, it is essential to ensure they function fairly across diverse user demographics. In this work, we show that LLMs suffer from personalization bias, where their performance is impacted when they are personalized to a user’s identity. We quantify personalization bias by evaluating the performance of LLMs along two axes - safety and utilty. We measure safety by examining how benign LLM responses are to unsafe prompts. We measure utility by evaluating the LLM’s performance on various tasks, including general knowledge, mathematical abilities, programming, and reasoning skills. We find that various LLMs, ranging from open-source models like Llama-3.1 (AI@Meta, 2024) and Mistral (Jiang et al., 2023) to API-based ones like GPT-3.5 (Ouyang et al., 2022) and GPT-4o (Achiam et al., 2023), exhibit significant variance in performance in terms of safety and utility when personalized with different user identities. Finally, we discuss several strategies to mitigate personalization bias and investigate the origin of personalization bias.

## Installation
The simplest way to run our implementation is to create with a new conda environment.

```
conda create -n pb python=3.9
source activate pb
pip install -r requirements.txt
```

## Data

We have provided the datasets (MMLU, strong_reject, do_not_answer) used in our experiments in the `data/` folder. Please cite the original sources of these datasets if you are using them in your work.



## Running Personalized LLMs

For open-source LLMs, run the command below
```
cd src/
python opensource.py --dataset [dataset_name] --model [model_name]
```

The dataset name can be `[mmlu, gsm8k, mbpp, dna, strongreject]`. The model names can be any of the keys in `model_dict` in `opensource.py`. 



For API-based LLMs, run the command below
```
cd src/
python api.py --dataset [dataset_name] --model [model_name]
```
The model name can be anything that OpenAI API supports. We perform experiments with two models `gpt-3.5-turbo-0125` and `gpt-4o`. 

After running the above commands, the LLM responses would be saved in the following path: `results/[dataset_name]/[model_name]/`.

## Evaluation

For evaluating utility-based datasets, run the command below for evaluation:
```
cd src/
python evaluate_utility.py --dataset [dataset_name] --model [model_name]
```

For evaluating safety-based datasets, run the command below for evaluation:
```
cd src/
python evaluate_safety.py --dataset [dataset_name] --model [model_name]
```

After running the above commands, the evaluation results would be saved in the following path: `results/[dataset_name]/[model_name]/results.json`.

## Running DPO-based Mitigation

To DPO-train a model using user identity-based system prompts, run the following command:
```
cd src/
python dpo.py --config [config_name]
```

The config name can be `[dpo, dpo-vanilla, dpo-random]`. `dpo` uses the original dataset for DPO training. `dpo-vanilla` performs DPO training without using system prompts. `dpo-random` perform DPO training by selecting a random user identity based system prompt for each input pair. 

## Reference


```
@inproceedings{vijjini2025exploring,
  title={Exploring Safety-Utility Trade-Offs in Personalized Language Models},
  author={Anvesh Rao Vijjini* and
          Somnath Basu Roy Chowdhury* and 
          Snigdha Chaturvedi},
  booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
  year={2025},
  url={https://openreview.net/forum?id=rtmuo0bGAb}
}
```