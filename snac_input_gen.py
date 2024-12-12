dsn = "amuvarma/snac-2m-raw"
push_name = "amuvarma/snac-2m-tts-combined"

from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
import string
import random

ds = load_dataset(dsn, split='train')

tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1 
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7

start_of_system = tokeniser_length + 8
end_of_system = tokeniser_length + 9
audio_tokens_start = tokeniser_length + 10

num_threads = os.cpu_count() - 2

def read_instructions(filename):
    instructions = []
    with open(filename, 'r') as file:
        instructions = file.read().splitlines()
    return instructions

instruction_list = read_instructions('read_out.txt')

def create_input_ids(example):
    random_instruction = random.choice(instruction_list)
    tokenized = tokeniser(random_instruction + " " + example["transcript"])
    tokenized_text = tokenized['input_ids'] + [end_of_text]
    codes = example["codes"]

    input_ids = (
        [start_of_human]
        + tokenized_text
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + codes
        + [end_of_speech]
        + [end_of_ai]
    )

    return {"input_ids": input_ids}

ds = ds.map(create_input_ids, num_proc=num_threads)
ds = ds.remove_columns(['transcript', 'codes'])


from datasets import Dataset
from tqdm import tqdm

print("Combining token lists...")
def combine_token_lists(dataset, num_threads=None):
    usable_length = (len(dataset) // 5) * 5
    if usable_length < len(dataset):
        print(f"Warning: Dataset length ({len(dataset)}) is not divisible by 5. Using first {usable_length} rows.")
    
    dataset = dataset.select(range(usable_length))
    
    def combine_batch(examples, indices):
        combined_ids = []
        for i in range(0, len(indices), 5):
            if i + 5 <= len(indices):
                combined = sum([examples['input_ids'][i + j] for j in range(5)], [])
                combined_ids.append(combined)
        return {'input_ids': combined_ids}

    return dataset.map(
        combine_batch,
        batched=True,
        with_indices=True,
        batch_size=1000,
        num_proc=num_threads,
        desc="Combining tokens",
        remove_columns=dataset.column_names, 
        
    )
ds = combine_token_lists(ds)

def pad_crop(example, max_length=8192):
    arr = example["input_ids"]
    if len(arr) > max_length:
        arr = arr[:max_length]
    if len(arr) < max_length:
        arr = arr + [pad_token]*(max_length - len(arr))
        
    attention_mask = [1 if token != pad_token else 0 for token in arr]
    labels = [token if token != pad_token else -100 for token in arr]

    return {
        "input_ids": arr,
        "attention_mask": attention_mask,
        "labels": labels
    }

ds = ds.map(pad_crop, num_proc=num_threads)

# Push to the hub
ds.push_to_hub(push_name)
