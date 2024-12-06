from datasets import load_dataset
dsn = "gpt-omni/VoiceAssistant-400K"
ds = load_dataset(dsn)
ds = ds["train"].remove_columns(['question', 'question_audio', 'answer',])
ds = ds.shuffle(seed=42)
dsnd = ds.select(range(10000))

# print(dsnd)
# dsnd.push_to_hub("amuvarma/voice-assistant-10k-processed-1")