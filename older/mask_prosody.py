from datasets import load_dataset
ds = load_dataset("amuvarma/6-layer-crossmodal-750k-llama-tts-0")

def preprocess_function(examples):
    pad_token = 0  # Assuming 0 is still your pad token
    start_ignore = 128266
    end_ignore = start_ignore + 1024

    def process_sequence(seq):
        return [
            -100 if (token_id == pad_token or start_ignore <= token_id <= end_ignore)
            else token_id
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

padded_ds = ds.map(preprocess_function, batched=True)

padded_ds.push_to_hub("amuvarma/audio_content_prosody_part1_470k_0")