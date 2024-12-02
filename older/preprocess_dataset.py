import torch
import os



from transformers import AutoTokenizer

print("cpu count",os.cpu_count())
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token


def preprocess_dataset(ds):

    ds = ds.remove_columns([col for col in ds.column_names if col not in ["transcript", "audio"]])

    instruction = "Tell me the exact phrase I say back, my phrase is: "
    instruction_inputs = tokenizer(instruction)
    instruction_input_ids = instruction_inputs["input_ids"]

    post_instruction = "The phrase you said is: "
    post_instruction_inputs = tokenizer(post_instruction)
    post_instruction_input_ids = post_instruction_inputs["input_ids"]

    
    def process_example(example):
        example["audio_values"] = torch.tensor(example["audio"]["array"])
        example["transcript_ids"] = tokenizer(example["transcript"])["input_ids"]
        example["labels"] =  example["transcript_ids"]
        example["input_ids"] = instruction_input_ids
        example["post_instruction"] = post_instruction_input_ids 
        return example

    ds = ds.map(process_example)
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["audio_values", "transcript_ids", "labels", "input_ids" ]])
    
    return ds
