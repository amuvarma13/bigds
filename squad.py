import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import random

tkn = "meta-llama/Llama-3.2-3B-Instruct"
dsn = "rajpurkar/squad"

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

start_of_system = tokeniser_length + 8
end_of_system = tokeniser_length + 9

audio_tokens_start = tokeniser_length + 10


ds = load_dataset(dsn, split='train')



def create_context_tokens(example):
    text_tokens = tokenizer.encode(example['context'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'context_text': text_tokens}

ds_txt = ds.map(
    create_context_tokens,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_question_tokens(example):
    
    text_tokens = tokenizer.encode(example['question'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'question_text': text_tokens}

ds_txt = ds_txt.map(
    create_question_tokens,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_answers_tokens(example):


    text_tokens = tokenizer.encode(example['answers']["text"][0], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'answer_text': text_tokens}

ds_txt = ds_txt.map(
    create_answers_tokens,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)


tts_dataset = ds_txt



def create_input_ids(example):
    input_ids = (
        [start_of_system] +
        example['context_text'] +
        [end_of_system]+
        [start_of_human] +
        example['question_text'] +
        [end_of_human] +
        [start_of_ai] +
        example['answer_text'] +
        [end_of_ai] 
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

all_columns = full_processed_padded.column_names

# Specify the columns we want to keep
columns_to_keep = ["input_ids", "attention_mask", "labels"]

# Identify columns to remove
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

# Remove unwanted columns
dataset_to_upload = full_processed_padded.remove_columns(columns_to_remove)

# Now upload the dataset with only the desired columns
dataset_to_upload.push_to_hub("amuvarma/squad-150k")