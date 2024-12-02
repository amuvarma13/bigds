from datasets import load_dataset
ds = load_dataset("amuvarma/500k-wdups-tts-0")

def preprocess_function(examples):
    start_ignore = 256010
    end_ignore = start_ignore + 1024

    def process_sequence(seq):
        return [
            token_id if (start_ignore <= token_id <= end_ignore) else -100
            for token_id in seq
        ]

    # Check if input is a single example or a batch
    if isinstance(examples['input_ids'], list) and isinstance(examples['input_ids'][0], list):
        # Batched input
        examples['labels'] = [process_sequence(seq) for seq in examples['input_ids']]
    else:
        # Single example
        examples['labels'] = process_sequence(examples['input_ids'])

    return examples

padded_ds = ds.map(
    preprocess_function, 
    num_proc=88,
    batched=True)

padded_ds.push_to_hub("amuvarma/500k-wdups-tts-0ablate-content_0")