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

# Tokenize function: remove punctuation, lowercase, and prepend/append special tokens
def tokenize_fn(example):
    # Remove punctuation and lowercase prompt and response texts

    random_instruction_1 = random.choice(instruction_list)
    random_instruction_2 = random.choice(instruction_list)
    prompt_text = example["text_prompt"].translate(str.maketrans('', '', string.punctuation)).lower()
    response_text = example["text_response"].translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Here we simply tokenize without adding extra instructions.
    # Prepend start_of_text and append end_of_text for both sequences.
    prompt_ids = [start_of_text] + tokenizer.encode(random_instruction_1 + " " + prompt_text, add_special_tokens=True) + [end_of_text]
    response_ids = [start_of_text] + tokenizer.encode(random_instruction_2 + " " + response_text, add_special_tokens=True) + [end_of_text]
    
    example["prompt_tokens"] = prompt_ids
    example["response_tokens"] = response_ids
    return example

ds = ds.map(tokenize_fn, num_proc=num_proc, desc="Tokenizing")

# Filter out rows missing required codes list fields
ds = ds.filter(lambda x: x.get("codes_list_prompt") is not None and x.get("codes_list_response") is not None)

# Create input_ids and compute labels. Here we include every token in the response segments,
# meaning all special tokens in Segment 2 and Segment 3 are used for computing loss.
def create_input_ids(example):
    # Build input_ids by concatenating segments:
    # Segment 0: [start_of_human] + prompt_tokens + [end_of_human]
    # Segment 1: [start_of_ai] + [start_of_speech] + codes_list_prompt + [end_of_speech] + [end_of_ai]
    # Segment 2: [start_of_human] + response_tokens + [end_of_human]
    # Segment 3: [start_of_ai] + [start_of_speech] + codes_list_response + [end_of_speech]
    input_ids = (
        [start_of_human] +
        example["prompt_tokens"] +
        [end_of_human] +
        [start_of_ai] +
        [start_of_speech] +
        example["codes_list_prompt"] +
        [end_of_speech] +
        [end_of_ai] +
        [start_of_human] +
        example["response_tokens"] +
        [end_of_human] +
        [start_of_ai] +
        [start_of_speech] +
        example["codes_list_response"] +
        [end_of_speech]
    )
    example["input_ids"] = input_ids

    # Initialize labels with -100 (ignore index)
    labels = [-100] * len(input_ids)

    # Calculate segment lengths:
    segment0_len = 1 + len(example["prompt_tokens"]) + 1  # [start_of_human] + prompt_tokens + [end_of_human]
    segment1_len = 1 + 1 + len(example["codes_list_prompt"]) + 1 + 1  # [start_of_ai] + [start_of_speech] + codes_list_prompt + [end_of_speech] + [end_of_ai]
    segment2_len = 1 + len(example["response_tokens"]) + 1  # [start_of_human] + response_tokens + [end_of_human]
    segment3_len = 1 + 1 + len(example["codes_list_response"]) + 1  # [start_of_ai] + [start_of_speech] + codes_list_response + [end_of_speech]

    # Label every token in Segment 2 (the text response, including its special tokens)
    segment2_start = segment0_len + segment1_len
    for i in range(segment2_start, segment2_start + segment2_len):
        labels[i] = input_ids[i]

    # Label every token in Segment 3 (the codes_list_response, including its special tokens)
    segment3_start = segment0_len + segment1_len + segment2_len
    for i in range(segment3_start, segment3_start + segment3_len):
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
