from datasets import load_dataset
dsn = "gpt-omni/VoiceAssistant-400K"
ds = load_dataset(dsn)

dsnd = ds["train"].select(range(10000))

print(dsnd)
# dsnd.push_to_hub("amuvarma/voice-assistant-10k-processed-1")