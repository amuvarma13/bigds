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

# Load dataset
ds = load_dataset(dsn, split='train')

num_proc = os.cpu_count() - 2

# Tokenize function with punctuation removal and lowercasing
def tokenize_fn(example):
    # Remove punctuation and lower-case for both prompt and response texts
    prompt_text = example["text_prompt"].translate(str.maketrans('', '', string.punctuation)).lower()
    response_text = example["text_response"].translate(str.maketrans('', '', string.punctuation)).lower()
    
    random_instruction_prompt = random.choice(instruction_list)
    random_instruction_response = random.choice(instruction_list)
    
    # prompt_ids = tokenizer.encode(random_instruction_prompt + " " + prompt_text, add_special_tokens=True)
    # response_ids = tokenizer.encode(random_instruction_response + " " + response_text, add_special_tokens=True)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    response_ids = tokenizer.encode(response_text, add_special_tokens=True)
    
    prompt_ids.append(end_of_text)
    response_ids.append(end_of_text)
    
    example["prompt_tokens"] = prompt_ids
    example["response_tokens"] = response_ids
    return example

ds = ds.map(tokenize_fn, num_proc=num_proc, desc="Tokenizing")

# Filter out rows missing required codes list fields
ds = ds.filter(lambda x: x.get("codes_list_prompt") is not None and x.get("codes_list_response") is not None)

# Create input_ids and compute labels for both the second text response and the codes_list_response.
def create_input_ids(example):
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

    # Initialize all labels with ignore index (-100)
    labels = [-100] * len(input_ids)

    # Compute lengths for each segment
    segment0_len = 1 + len(example["prompt_tokens"]) + 1  # [start_of_human] + prompt_tokens + [end_of_human]
    segment1_len = 1 + 1 + len(example["codes_list_prompt"]) + 1 + 1  # [start_of_ai] + [start_of_speech] + codes_list_prompt + [end_of_speech] + [end_of_ai]
    segment2_len = 1 + len(example["response_tokens"]) + 1  # [start_of_human] + response_tokens + [end_of_human]
    # Segment3 is: [start_of_ai] + [start_of_speech] + codes_list_response + [end_of_speech]

    # Label the second text response tokens (in segment2)
    segment2_start = segment0_len + segment1_len
    # Skip the first token of segment2 ([start_of_human])
    response_tokens_start = segment2_start + 1
    response_tokens_length = len(example["response_tokens"])
    response_tokens_end = response_tokens_start + response_tokens_length

    for i in range(response_tokens_start, response_tokens_end):
        labels[i] = input_ids[i]

    # Label the codes_list_response tokens (in segment3)
    segment3_start = segment0_len + segment1_len + segment2_len
    # Skip the first two tokens of segment3 ([start_of_ai] and [start_of_speech])
    codes_start = segment3_start + 2
    codes_length = len(example["codes_list_response"])
    codes_end = codes_start + codes_length

    for i in range(codes_start, codes_end):
        labels[i] = input_ids[i]

    example["labels"] = labels
    example["attention_mask"] = [1] * len(input_ids)
    return example

ds = ds.map(create_input_ids, num_proc=num_proc, desc="Creating input IDs")

# Keep only the desired columns for pushing
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
ds = ds.remove_columns(columns_to_remove)
print(ds.column_names)

ds.push_to_hub(push_name)
