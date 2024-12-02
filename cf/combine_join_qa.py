dsns = ["amuvarma/qa_large_0_0_speechq", "amuvarma/qa_large_0_2_speechq", "amuvarma/qa_large_0_3_speechq", "amuvarma/qa_large_0_5_speechq"]

from datasets import load_dataset, concatenate_datasets
full_dataset =  concatenate_datasets([load_dataset(dsn)["train"] for dsn in dsns])
dsqaog = load_dataset("amuvarma/qa_1_4_new")
dsqaog = dsqaog["train"].rename_column("transformed_question", "audio")
dsqaog = dsqaog.rename_column("question", "user")
dsqaog = dsqaog.rename_column("answer", "assistant")


full_dataset = full_dataset.rename_column("answer", "assistant")
full_dataset = full_dataset.rename_column("question", "user")

dsf = concatenate_datasets([dsqaog, full_dataset])

dsm = "amuvarma/mls-eng-10k-500k-projection_prep"
dsmm = load_dataset(dsm)["train"]

dsmm = dsmm.remove_columns(["transcript"])


# Speed up mapping with num_proc and batched processing
from datasets import load_dataset, concatenate_datasets, Audio
# Align the audio feature in dsmm
dsmm = dsmm.cast_column("audio", Audio(sampling_rate=16000))

# Align the audio feature in dsf
dsf = dsf.cast_column("audio", Audio(sampling_rate=16000))


# Now they should concatenate without sampling rate issues
dssf = concatenate_datasets([dsmm, dsf])

dssf = dssf.shuffle(seed=42)

dssf.push_to_hub("amuvarma/proj-train-qa-and-speechqa")