"""Created: 2022-03-25
GitHub Repo: https://github.com/VictorieeMan/Basic_T5_Transformer.git
An attempt at creating an easy to mod and use T5 Transformer Model for the PyTorch framework.
"""

""" To-do and doing list:
Example:
*VE reading up on T5

"""

try:
    import os
    import json
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import transformers

    from transformers import AutoTokenizer, AutoModelWithLMHead

    from tqdm import tqdm # Progressbar
    # *https://www.pythonpool.com/python-tqdm/

except Exception as e:
    raise Exception("Couldn't load all modules. More info: {}".format(e))

# print("Gooday ax shaft!")

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelWithLMHead.from_pretrained("t5-small")

#Hyperparameters
max_source_len = 512
max_target_len = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available device:",device)

#Training examples
input_seq_1 = "There's a monkey under the table."
output_seq_1 = "Det är en apa under bordet."

input_seq_2 = "Why do pigs fly?"
output_seq_2 = "Varför flyger grisar?"

#Encoding the inputs
task_prefix = "Translate Eng to Swe"
input_sequences = [input_seq_1,input_seq_2]
encoding = tokenizer(
    [task_prefix+ seq for seq in input_sequences],
    padding = "longest",
    max_length=max_source_len,
    truncation=True,
    return_tensors="pt"
)
input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

#Encoding targets
target_encoding = tokenizer(
    [output_seq_1, output_seq_2], padding="longest", max_length=max_target_len, truncation=True
)
labels = target_encoding.input_ids

#Replace padding token id's of the labels
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
