import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import random

tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tkn)
# dsn = "amuvarma/6-interleave-800k-0"
dsn = "amuvarma/750k-raw_dups3-0"
total_examples = 748000
mid_point = total_examples



cpu_count = multiprocessing.cpu_count()

# Set num_threads to CPU count or a maximum of 8, whichever is smaller
num_threads = cpu_count

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

audio_tokens_start = tokeniser_length + 10


ds = load_dataset(dsn)

def create_audio_tokens(example):
    audio_tokens = []
    max_length = len(example['facodec_0'])

    # Define the new order of columns for processing
    column_order = [1, 0, 2, 3, 4, 5]

    for j in range(max_length):
        for i, original_i in enumerate(column_order):
            offset = audio_tokens_start + (i * 1024)  # Offset based on position in column_order
            facodec_column = f'facodec_{original_i}'  # Use original index for column name
            modified_token = example[facodec_column][j] + offset
            audio_tokens.append(modified_token)

    return {'audio_tokens': audio_tokens}


# Apply the function to create the new column using multithreading
ds_aud = ds['train'].map(
    create_audio_tokens,
    num_proc=num_threads,  # Use the globally defined num_threads
    desc="Creating audio_tokens column"  # Optional: adds a progress bar description
)

def create_text_tokens(example):
    text_tokens = tokenizer.encode(example['transcript'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'text_tokens': text_tokens}

ds_txt = ds_aud.map(
    create_text_tokens,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def load_file_to_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


stt_list = load_file_to_list("stt.txt")


tts_list = load_file_to_list("tts.txt")

tts_dataset = ds_txt.select(range(mid_point))
stt_dataset = ds_txt.select(range(mid_point, total_examples))

import random
from functools import partial

def process_tts_row(example, tts_list, tokenizer):
    random_string = random.choice(tts_list)

    if random.random() < 0.7:
        random_string += '\n'

    # combined_text = f"{random_string}{example['transcript']}"
    combined_text = f"{example['transcript']}"


    tokens = tokenizer.encode(combined_text, add_special_tokens=True)

    tokens.append(end_of_text)

    example['system_message'] = tokens

    return example

process_row_partial = partial(process_tts_row, tts_list=tts_list, tokenizer=tokenizer)

tts_dataset = tts_dataset.map(
    process_row_partial,
    num_proc=num_threads,  # Use the globally defined num_threads
    desc="Processing TTS dataset"
)

def create_input_ids(example):
    input_ids = (
        [start_of_human] +
        example['system_message'] +
        [end_of_human, start_of_ai, start_of_speech] +
        example['audio_tokens'] +
        [end_of_speech, end_of_ai]
    )

    example['input_ids'] = input_ids
    return example

# Apply the function to the TTS dataset using multiple threads
tts_dataset = tts_dataset.map(
    create_input_ids,
    num_proc=num_threads,
    desc="Creating input_ids column"
)



def create_input_ids_stt(example):
    # Choose and tokenize a random string from stt_list
    random_string = random.choice(stt_list)
    if random.random() < 0.7:  # Add a line break 70% of the time
        random_string += '\n'
    tokenized_random_string = tokenizer.encode(random_string, add_special_tokens=True)

    # Construct the input_ids
    input_ids = (
        [start_of_human] +
        tokenized_random_string +
        [end_of_text,
         start_of_speech] +
        example['audio_tokens'] +
        [end_of_speech, end_of_human, start_of_ai] +
        example['text_tokens'] +
        [end_of_ai]
    )

    example['input_ids'] = input_ids
    return example

# Apply the function to the STT dataset using multiple threads
stt_dataset = stt_dataset.map(
    create_input_ids_stt,
    num_proc=num_threads,
    desc="Creating input_ids column for STT dataset"
)


columns_to_keep = ['transcript', 'input_ids'] + [f'facodec_{i}' for i in range(6)]


# Function to keep only specified columns
def keep_columns(dataset, columns):
    return dataset.remove_columns([col for col in dataset.column_names if col not in columns])

# Keep only the specified columns in both datasets
tts_dataset = keep_columns(tts_dataset, columns_to_keep)
stt_dataset = keep_columns(stt_dataset, columns_to_keep)

# Combine the datasets
combined_dataset = concatenate_datasets([tts_dataset, stt_dataset])

# Shuffle the combined dataset
combined_dataset = combined_dataset.shuffle(seed=42)


max_length = 8192


def pad_and_create_mask(example):
    # Pad or truncate input_ids
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
        example['attention_mask'] = [1] * max_length
    else:
        padding_length = max_length - len(example['input_ids'])
        example['attention_mask'] = [1] * len(example['input_ids']) + [0] * padding_length
        example['input_ids'] = example['input_ids'] + [pad_token] * padding_length

    return example

# Apply padding and create attention mask
full_processed_padded = combined_dataset.map(
    pad_and_create_mask,
    num_proc=88  # Adjust based on your CPU cores
)

def preprocess_function(examples, ):
    examples['labels'] = [
        (token_id if token_id != pad_token else -100) for token_id in examples['input_ids']
    ]
    return examples

full_processed_padded = full_processed_padded.map(
    preprocess_function,
    num_proc=88
)

full_processed_padded.push_to_hub("amuvarma/6-layer-crossmodal-750k-llama-tts-0")
# full_processed_padded.push_to_hub("amuvarma/6-layer-crossmodal-1k-5")