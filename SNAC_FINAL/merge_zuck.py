from datasets import load_dataset
import os

dsn = "amuvarma/va-310k-320k-snac"
ds = load_dataset(dsn)

ds_l1n = "eliasfiz/zucktts-0_audio"
ds_l2n = "eliasfiz/zucktts-1_audio"
ds_l3n = "eliasfiz/zucktts-2_audio"

ds_l1 = load_dataset(ds_l1n,split="train")
ds_l2 = load_dataset(ds_l2n,split="train")
ds_l3 = load_dataset(ds_l3n,split="train")

ds = ds["train"]
ds1 = ds.select(range(0,2000))
ds2 = ds.select(range(2000, 4000))
ds3 = ds.select(range(4000, 6000))

def add_audio_column(batch, indices):
    # indices is a list of row indices for this batch
    # ds_l1[indices]["answer_audio"] gets only the rows you need
    return {"answer_audio": ds_l1[indices]["answer_audio"]}

cds1 = ds1.map(
    add_audio_column,
    with_indices=True,
    batched=True,
    batch_size=1,  # you can tune this to fit RAM constraints
    num_proc=os.cpu_count() - 2
)


def add_audio_column2(batch, indices):
    # indices is a list of row indices for this batch
    # ds_l1[indices]["answer_audio"] gets only the rows you need
    return {"answer_audio": ds_l2[indices]["answer_audio"]}

cds2 = ds2.map(
    add_audio_column2,
    with_indices=True,
    batched=True,
    batch_size=1,
    num_proc=os.cpu_count() - 2
        # you can tune this to fit RAM constraints
)
def add_audio_column3(batch, indices):
    # indices is a list of row indices for this batch
    # ds_l1[indices]["answer_audio"] gets only the rows you need
    return {"answer_audio": ds_l3[indices]["answer_audio"]}

cds3 = ds3.map(
    add_audio_column3,
    with_indices=True,
    batched=True,
    batch_size=1, 
    num_proc=os.cpu_count() - 2
        # you can tune this to fit RAM constraints
)


from datasets import Audio

from datasets import concatenate_datasets
tds = concatenate_datasets([cds1, cds2, cds3])

tds = tds.cast_column("question_audio", Audio(sampling_rate=16000))
tds = tds.cast_column("answer_audio", Audio(sampling_rate=24000))
tds.push_to_hub("amu-zucktts-with-qaudio-total-cast")