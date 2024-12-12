from datasets import load_dataset, concatenate_datasets

dsn1 = "CanopyLabs/audio_pretrain_10m-facodec"
dsn2 = "CanopyElias/audio_pretrain_10m-facodec"
dsn3 = "eliasfiz/audio_pretrain_10m-facodec"

ds1 = load_dataset(dsn1, split='train')
ds2 = load_dataset(dsn2, split='train')
ds3 = load_dataset(dsn3, split='train')


ds_full = concatenate_datasets([ds1, ds2, ds3])

print(ds_full)

ds_full.remove_columns_(["audio"])

ds_full.push_to_hub("amuvarma/audio_pretrain_10m")