import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import random

tkn = "meta-llama/Llama-3.2-3B-Instruct"
dsn = "amuvarma/va-320k-330k-snac-no-identity"
pushname = "amuvarma/va-320k-330k-snac-no-identity-QA_TTTTS"

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





def create_question_tokens(example):
    
    text_tokens = tokenizer.encode(example['question'], add_special_tokens=True)
    text_tokens.append(end_of_text)  # Append token 1 to the end
    return {'question_text': text_tokens}

ds_txt = ds.map(
    create_question_tokens,
    num_proc=num_threads,
    desc="Creating text_tokens column"
)

def create_answers_tokens(example):


    text_tokens = tokenizer.encode(example['answer'], add_special_tokens=True)
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

        [start_of_human] +
        example['question_text'] +
        [end_of_human] +
        [start_of_ai] +
        example['answer_text']
    )

    example['input_ids'] = input_ids
    example["attention_mask"] = [1] * len(input_ids)
    example["labels"] = input_ids
    return example

tts_dataset = tts_dataset.map(
    create_input_ids,
    num_proc=num_threads,
    desc="Creating input_ids column"
)


max_length = 8192

columns_to_keep = ["input_ids", "attention_mask", "labels"]

# Identify columns to remove
all_columns = tts_dataset.column_names
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

# Remove unwanted columns
dataset_to_upload = tts_dataset.remove_columns(columns_to_remove)

# Now upload the dataset with only the desired columns
dataset_to_upload.push_to_hub(pushname)