## TAKES IN DATASET WITH COLUMNS codes_list, question, answer

dsn = "amuvarma/snacced-flat-zuck-convo-sttsed"
push_name = "amuvarma/snacced-flat-zuck-convo-sttsed-proc"


from datasets import load_dataset, Dataset
import os
from transformers import AutoTokenizer
ds = load_dataset(dsn, split='train')
from collections import defaultdict


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

tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})


# Remove all columns except "answer_snac"
# columns_to_remove = [col for col in ds.column_names if col != "answer_snac"]
# ds = ds.remove_columns(columns_to_remove)

# Determine number of processes based on CPU count
num_proc = os.cpu_count() - 2

#filter out all rows without question, answer, or codes_list 
ds = ds.filter(lambda x: x['question'] and x['answer'] and x['codes_list'])

#filter out all rows with codeslist length over 12000
ds = ds.filter(lambda x: len(x['codes_list']) < 8192)

def dataset_to_list_of_lists(dataset):
    
    # Group rows by conversation_index
    conv_dict = defaultdict(list)
    
    # Populate the dictionary: {conversation_index: [rows]}
    for row in dataset:
        conv_index = row["conversation_index"]
        conv_dict[conv_index].append({
            "messages_index": row["messages_index"],
            "question": row["question"],
            "answer": row["answer"],
            "codes_list": row["codes_list"],
            "answer_audio": row["answer_audio"]
        })
    
    # Sort each conversation by messages_index, then collect into a list
    result = []
    for conv_index in sorted(conv_dict.keys()):
        messages_sorted = sorted(conv_dict[conv_index], key=lambda x: x["messages_index"])
        result.append(messages_sorted)
    
    return result

mylists = dataset_to_list_of_lists(ds)

print(len(mylists))




all_input_ids = []
all_audios = []
for convo in mylists:
    input_ids = []
    audios = []
    for message in convo:
        question = "<|audio|>"
        answer = message["answer"]
        codes_list = message["codes_list"]
        tokenised_question = tokenizer.encode(question, add_special_tokens=True)
        tokenised_answer = tokenizer.encode(answer, add_special_tokens=True)
        tokenised_question.append(end_of_text)
        tokenised_answer.append(end_of_text)
        input_ids.extend([start_of_human] + tokenised_question + [end_of_human] + [start_of_ai] + tokenised_answer + [start_of_speech] + codes_list + [end_of_speech] + [end_of_ai])
        audios.append(message["answer_audio"]["array"])

    all_input_ids.append(input_ids)
    all_audios.append(audios)
    
        
print(len(all_input_ids), len(all_audios))

def convert_to_hf_dataset(all_input_ids):
    flat_input_ids = [iids for iids in all_input_ids]
    ds = Dataset.from_dict({"input_ids": flat_input_ids, "audios": all_audios})
    return ds

ds = convert_to_hf_dataset(all_input_ids)

print(ds)

#add the labels and attention mask

def create_mask_and_labels(example):
    max_length = 8192

    if len(example['input_ids']) > max_length:
        example['attention_mask'] = [1] * max_length
        example['input_ids'] = example['input_ids'][:max_length]
    else:
        example['attention_mask'] = [1] * len(example['input_ids'])

    example['labels'] = example['input_ids']
    
    return example

ds = ds.map(create_mask_and_labels, num_proc=num_proc)


columns_to_keep = ["input_ids", "labels",   "attention_mask", "audios"]

columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)

ds.push_to_hub(push_name)   





# # Map the function in parallel
# def tokenize_fn(example):
#     user_ids = tokenizer.encode(example["question"], add_special_tokens=True)
#     answer_ids = tokenizer.encode(example["answer"], add_special_tokens=True)
#     user_ids.append(end_of_text)
#     answer_ids.append(end_of_text)
#     example["user_tokens"] = user_ids
#     example["answer_tokens"] = answer_ids
#     return example

# ds = ds.map(tokenize_fn, num_proc=num_proc)

# def create_input_ids(example):
#     input_ids = (
#         [start_of_human]
#         + example["user_tokens"]
#         + [end_of_human]
#         + [start_of_ai]
#         + example["answer_tokens"]
#         + [start_of_speech]
#         + example["codes_list"]
#         + [end_of_speech]
#         + [end_of_ai]
#     )
#     example["input_ids"] = input_ids
#     example["labels"] = input_ids
#     example["attention_mask"] = [1] * len(input_ids)
#     return example

# ds = ds.map(create_input_ids, num_proc=num_proc)

# columns_to_keep = ["input_ids", "labels",   "attention_mask"]
# columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

# ds = ds.remove_columns(columns_to_remove)
# print(ds.column_names)


# ds.push_to_hub(push_name)