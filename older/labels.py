from datasets import load_dataset

full_processed_padded = load_dataset("amuvarma/6_layer_interleave-102345-500k-0")

pad_token = 0

def preprocess_function(examples, ):
    examples['labels'] = [
        (token_id if token_id != pad_token else -100) for token_id in examples['input_ids']
    ]
    return examples

full_processed_padded = full_processed_padded.map(
    preprocess_function,
    num_proc=88
)

full_processed_padded.push_to_hub("amuvarma/6_layer_interleave-102345-500k-1")