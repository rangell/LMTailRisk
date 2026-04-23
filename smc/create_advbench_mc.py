import json
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


path = Path("./harmful_behaviors.csv")

data = pd.read_csv(path)


from datasets import load_dataset

# model_shortname = "Qwen2.5-7B-Instruct"
model_shortname = "Llama-3.1-8B-Instruct"
# model_shortname = 'phi-4'

old_mc = load_dataset(
    "json",
    data_files=f"/gpfs/data/ranganathlab/singhr36/rare-generations/monte_carlo_estimates/results/strong_reject/{model_shortname}_mc_est_10k.json",
)["train"]

sample_dict = {'category': None, 'source': None, 'forbidden_prompt': None, 'jailbroken_prompt': None, 'score': [0]}


mc_dataset = []
for idx, row in tqdm(data.iterrows(), total=len(data)):
    mc_dataset.append(sample_dict.copy())
    mc_dataset[idx]['forbidden_prompt'] = row['goal']

# # old_mc = load_dataset(
# #     "json",
# #     data_files="greedy_icml_unsafe/Llama-3.1-8B-Instruct_mc_est_10k.json",
# # )["train"]

# assert len(old_mc) == len(data) // 25 + 1, "The length of the old MC estimates should be equal to the number of unique indices in the new data divided by 25 (since each index has 25 paraphrased examples)."


# # indices_to_keep = np.random.choice(len(old_mc), size=100, replace=False)
# # indices_to_keep = np.arange(100)
# indices_to_keep = np.arange(100, 200)

# fake_mc_file = []
# for d in tqdm(data):
#     original_index = d['index']
#     if original_index not in indices_to_keep:
#         continue
    
#     old_mc_values = old_mc[original_index]
    
#     assert d['original_forbidden_prompt'] == old_mc_values['forbidden_prompt']
    
#     old_mc_values.pop("forbidden_prompt")
#     old_mc_values.pop("response")
        
#     combined_dict = {**d, **old_mc_values}
    
#     fake_mc_file.append(combined_dict)

with open(f'advbench_mc.json', 'w') as f:
    json.dump(mc_dataset, f, indent=4)

from IPython import embed

embed()
exit()
