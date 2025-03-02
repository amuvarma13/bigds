import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from huggingface_hub import snapshot_download


tkn = "google/gemma-2-9b-it"
dsn = "facebook/natural_reasoning"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)
pushname = "amuvarma/facebook-natural-reasoning-TTT"

tokenizer = AutoTokenizer.from_pretrained(tkn)

cpu_count = multiprocessing.cpu_count()

num_threads = cpu_count


tokeniser_length = 256000
start_of_text = 2
end_of_text = tokeniser_length + 8

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10


ds = load_dataset(dsn, split='train')


def create_input_ids(example):

        
    qtext_tokens = tokenizer.encode(example['question'], add_special_tokens=True)
    qtext_tokens.append(end_of_text)  # Append token 1 to the end

    atext_tokens = tokenizer.encode(example['responses']["response"], add_special_tokens=True)
    atext_tokens.append(end_of_text)  # Append token 1 to the end
    input_ids = (

        [start_of_human] +
        qtext_tokens+
        [end_of_human] +
        [start_of_ai] +
        atext_tokens
    )

    example['input_ids'] = input_ids
    return example

tts_dataset = ds.map(
    create_input_ids,
    num_proc=num_threads,
    desc="Creating input_ids column", 
    remove_columns=ds.column_names
)




columns_to_keep = ["input_ids"]

# Identify columns to remove
all_columns = tts_dataset.column_names
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

# Remove unwanted columns
dataset_to_upload = tts_dataset.remove_columns(columns_to_remove)

# Now upload the dataset with only the desired columns
dataset_to_upload.push_to_hub(pushname)