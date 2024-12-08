from datasets import load_dataset
dsn = "gpt-omni/VoiceAssistant-400K"
ds = load_dataset(dsn)
ds = ds["train"].remove_columns(['split_name', 'index', 'round', 'answer_snac'])
ds = ds.shuffle(seed=42)
# dsnd = ds.select(range(200000, 250000))
dsnd = ds.select(range(250000, 255000))

# print(dsnd)
dsnd.push_to_hub("amuvarma/voice-assistant-250-255k-processed")