import string
import random
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Dataset and tokenizer settings
dsn = "amuvarma/zst-snacced"
push_name = "amuvarma/zst-snacced-proc"
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

audio_tokens_start = tokeniser_length + 10

tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Load instructions from file
def read_instructions(filename):
    with open(filename, 'r') as file:
        instructions = file.read().splitlines()
    return instructions

instruction_list = read_instructions('read_out.txt')

# Load dataset and filter rows
ds = load_dataset(dsn, split='train')
# ds = ds.filter(lambda x: x['question'] and x['answer'] and x['codes_list'])
# ds = ds.filter(lambda x: len(x['codes_list']) < 8192)

num_proc = os.cpu_count() - 2

# Tokenize function with punctuation removal and lowercasing
def tokenize_fn(example):
    # Remove punctuation and convert to lowercase for both prompt and response texts
    prompt_text = example["text_prompt"].translate(str.maketrans('', '', string.punctuation)).lower()
    response_text = example["text_response"].translate(str.maketrans('', '', string.punctuation)).lower()
    
    random_instruction_prompt = random.choice(instruction_list)
    random_instruction_response = random.choice(instruction_list)
    
    prompt_ids = tokenizer.encode(random_instruction_prompt + " " + prompt_text, add_special_tokens=True)
    response_ids = tokenizer.encode(random_instruction_response + " " + response_text, add_special_tokens=True)
    
    prompt_ids.append(end_of_text)
    response_ids.append(end_of_text)
    
    example["prompt_tokens"] = prompt_ids
    example["response_tokens"] = response_ids
    return example

ds = ds.map(tokenize_fn, num_proc=num_proc)

# Create input_ids and compute labels only on the second sample’s codes list.
def create_input_ids(example):
    # Build input_ids by concatenating several segments
    input_ids = (
        [start_of_human]
        + example["prompt_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list_prompt"]
        + [end_of_speech]
        + [end_of_ai]
        + [start_of_human]
        + example["response_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list_response"]
        + [end_of_speech]
    )
    example["input_ids"] = input_ids

    # Initialize all labels to ignore index (-100)
    labels = [-100] * len(input_ids)

    # Calculate segment lengths
    segment0_len = 1 + len(example["prompt_tokens"]) + 1
    segment1_len = 1 + 1 + len(example["codes_list_prompt"]) + 1 + 1
    segment2_len = 1 + len(example["response_tokens"]) + 1

    # Segment3: tokens corresponding to codes_list_response
    # Segment3 structure: [start_of_ai] + [start_of_speech] + codes_list_response + [end_of_speech]
    # We want to compute loss only on codes_list_response tokens.
    segment3_start = segment0_len + segment1_len + segment2_len
    # The first two tokens of segment3 (start_of_ai, start_of_speech) are not labeled.
    codes_start = segment3_start + 2
    codes_length = len(example["codes_list_response"])
    codes_end = codes_start + codes_length

    # Copy the token IDs for the codes_list_response portion into labels
    for i in range(codes_start, codes_end):
        labels[i] = input_ids[i]

    example["labels"] = labels
    example["attention_mask"] = [1] * len(input_ids)
    return example

ds = ds.map(create_input_ids, num_proc=num_proc)

# Keep only the desired columns for pushing
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
ds = ds.remove_columns(columns_to_remove)
print(ds.column_names)

ds.push_to_hub(push_name)
