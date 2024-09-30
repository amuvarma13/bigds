import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import random

tkn = "google/gemma-2-2b"
dsn = "amuvarma/750k-raw"
tokeniser_length = 256000
pad_token = 0


tokenizer = AutoTokenizer.from_pretrained(tkn)

cpu_count = multiprocessing.cpu_count()

num_threads = cpu_count


start_of_text = 2
end_of_text = 1

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6


audio_tokens_start = tokeniser_length + 10

ds = load_dataset(dsn, split='train')


ds = ds[:500000]






def create_audio_tokens(example):
    audio_tokens = []
    max_length = len(example['facodec_0'])

    column_order = [1, 0, 2, 3, 4, 5]

    for j in range(max_length):
        for i, original_i in enumerate(column_order):
            offset = audio_tokens_start + (i * 1024)  # Offset based on position in column_order
            facodec_column = f'facodec_{original_i}'  # Use original index for column name
            modified_token = example[facodec_column][j] + offset
            audio_tokens.append(modified_token)

    return {'audio_tokens': audio_tokens}


# Apply the function to create the new column using multithreading
ds_aud = ds.map(
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







tts_dataset = ds_txt


def process_tts_row(example):
    combined_text = example['transcript']
    tokens = tokenizer.encode(combined_text, add_special_tokens=True)
    tokens.append(end_of_text)
    example['system_message'] = tokens
    return example


tts_dataset = tts_dataset.map(
    process_tts_row,
    num_proc=num_threads,
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

tts_dataset = tts_dataset.map(
    create_input_ids,
    num_proc=num_threads,
    desc="Creating input_ids column"
)


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


full_processed_padded = tts_dataset.map(
    pad_and_create_mask,
    num_proc=88 
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

full_processed_padded.push_to_hub("amuvarma/500k-wdups-tts-0")
