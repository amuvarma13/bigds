import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import random

tkn = "meta-llama/Llama-3.2-3B-Instruct"
dsn = "eliasfiz/merged_audio_conversational_facodec"
tokeniser_length = 256000



tokenizer = AutoTokenizer.from_pretrained(tkn)

cpu_count = multiprocessing.cpu_count()

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


ds = load_dataset(dsn, split='train')



def create_text_tokensh1(example):
    text_tokens = tokenizer.encode(example['human1'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'human1_text': text_tokens}

ds_txt = ds.map(
    create_text_tokensh1,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_text_tokensh2(example):
    text_tokens = tokenizer.encode(example['human2'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'human2_text': text_tokens}

ds_txt = ds_txt.map(
    create_text_tokensh2,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_text_tokensh3(example):
    text_tokens = tokenizer.encode(example['human3'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'human3_text': text_tokens}

ds_txt = ds_txt.map(
    create_text_tokensh3,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_text_tokensai1(example):
    text_tokens = tokenizer.encode(example['transcript1'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'transcript1_text': text_tokens}

ds_txt = ds_txt.map(
    create_text_tokensai1,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_text_tokensai2(example):
    text_tokens = tokenizer.encode(example['transcript2'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'transcript2_text': text_tokens}

ds_txt = ds_txt.map(
    create_text_tokensai2,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_text_tokensai3(example):
    text_tokens = tokenizer.encode(example['transcript3'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'transcript3_text': text_tokens}


ds_txt = ds_txt.map(
    create_text_tokensai3,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)



tts_dataset = ds_txt



def create_input_ids(example):
    input_ids = (
        [start_of_human] +
        example['human1_text'] +
        [end_of_human]+
        [start_of_ai] +
        example['transcript1_text'] +
        [end_of_ai] +
        [start_of_human] +
        example['human2_text'] +
        [end_of_human]+
        [start_of_ai] +
        example['transcript2_text'] +
        [end_of_ai] +
        [start_of_human] +
        example['human3_text'] +
        [end_of_human]+
        [start_of_ai] +
        example['transcript3_text'] +
        [start_of_speech] +
        [end_of_ai]
    )

    example['input_ids'] = input_ids
    return example

tts_dataset = tts_dataset.map(
    create_input_ids,
    num_proc=num_threads,
    desc="Creating input_ids column"
)


max_length = 250


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

all_columns = full_processed_padded.column_names

# Specify the columns we want to keep
columns_to_keep = ["input_ids", "attention_mask", "labels"]

# Identify columns to remove
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

# Remove unwanted columns
dataset_to_upload = full_processed_padded.remove_columns(columns_to_remove)

# Now upload the dataset with only the desired columns
dataset_to_upload.push_to_hub("amuvarma/conversation_text-tune-13k-25kl")