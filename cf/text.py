from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
import string



tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

push_name = "amuvarma/orcatext-dev-processed-1"

ds_name = "amuvarma/combined-shuffled-text-orca-ds"
ds = load_dataset(ds_name, split="train")

ds = ds.select(range(1000))
 

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

start_of_system = tokeniser_length + 8
end_of_system = tokeniser_length + 9

audio_tokens_start = tokeniser_length + 10


print(ds)

import json

def read_instructions(filename):
    instructions = []
    with open(filename, 'r') as file:
        instructions = file.read().splitlines()
    return instructions

instruction_list = read_instructions('text_instructs.txt')

import json
import random

def process_batch(batch):
    input_ids = []
    for messages_str in batch["messages"]:
        sequence = []
        instruction = random.choice(instruction_list)
        sequence.extend([start_of_system] + tokeniser(instruction)["input_ids"] + [end_of_text, end_of_system])
        
        messages = json.loads(messages_str)
        for msg in messages:
            if msg["role"] == "user":
                sequence.extend([start_of_human] + tokeniser(msg["content"])["input_ids"] + [end_of_text, end_of_human])
            elif msg["role"] == "assistant":
                sequence.extend([start_of_ai] + tokeniser(msg["content"])["input_ids"] + [end_of_text, end_of_ai])
        input_ids.append(sequence)
    return {"input_ids": input_ids}

ds1 = ds.map(
    process_batch,
    batched=True,
    batch_size=1000
)


max_length = 8192
def create_mask(example):
    if len(example['input_ids']) > max_length:
        example['attention_mask'] = [1] * max_length
        example['input_ids'] = example['input_ids'][:max_length]
    else:
        example['attention_mask'] = [1] * len(example['input_ids'])
    
    return example

ds_2 = ds1.map(create_mask)

print(ds_2)

def preprocess_function(examples):
    examples['labels'] = [
        [(token_id if token_id != pad_token else -100) for token_id in input_ids]
        for input_ids in examples['input_ids']
    ]
    return examples


num_cpus = os.cpu_count()

num_processes = max(1, int(num_cpus * 0.75))

ds_3 = ds_2.map(
    preprocess_function,
    batched=True,
    num_proc=num_processes,
    desc="Preprocessing dataset"
)


ds_3.push_to_hub(push_name)