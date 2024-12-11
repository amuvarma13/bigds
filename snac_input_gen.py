dsn = "amuvarma/snac-2m-raw"

push_name = "amuvarma/snac-2m-tts"

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
    input_ids = []

    random_instruction = random.choice(instruction_list)
    tokenized = tokeniser(random_instruction+ " "+example["transcript"])
    tokenized_text = tokenized['input_ids'] + [end_of_text]
    codes = example["codes"]

    input_ids = [start_of_human]+tokenized_text + [end_of_human]+[start_of_ai] + [start_of_speech] + codes + [end_of_speech] + [end_of_ai]

    return input_ids


#map the dataset
ds = ds.map(create_input_ids, num_proc=num_threads)

ds = ds.remove_columns_(['transcript', 'codes'])

#push to the hub
ds.push_to_hub(push_name)   
    