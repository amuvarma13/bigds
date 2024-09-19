from huggingface_hub import hf_hub_download
import os
from datasets import load_dataset_builder
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset

dataset_name = "eliasfiz/audio"
split = "train"
local_dir = "./downloaded_parquet_files"
files_to_download = parquet_files = [
    "vm_0_0-00000-of-00001.parquet",
    "vm_0_1-00000-of-00001.parquet",
    "vm_0_2-00000-of-00001.parquet",
    "vm_10_0-00000-of-00001.parquet",
    "vm_10_1-00000-of-00001.parquet",
    "vm_10_2-00000-of-00001.parquet",
    "vm_11_0-00000-of-00001.parquet",
    "vm_11_1-00000-of-00001.parquet",
    "vm_11_2-00000-of-00001.parquet",
    "vm_13_0-00000-of-00001.parquet",
    "vm_13_1-00000-of-00001.parquet",
    "vm_14_0-00000-of-00001.parquet",
    "vm_14_1-00000-of-00001.parquet",
    "vm_14_2-00000-of-00001.parquet",
    "vm_15_0-00000-of-00001.parquet",
    "vm_15_1-00000-of-00001.parquet",
    "vm_15_2-00000-of-00001.parquet",
    "vm_16_0-00000-of-00001.parquet",
    "vm_16_1-00000-of-00001.parquet",
    "vm_16_2-00000-of-00001.parquet",
    "vm_17_0-00000-of-00001.parquet",
    "vm_17_1-00000-of-00001.parquet",
    "vm_17_2-00000-of-00001.parquet",
    "vm_18_0-00000-of-00001.parquet",
    "vm_18_1-00000-of-00001.parquet",
    "vm_18_2-00000-of-00001.parquet",
    "vm_19_0-00000-of-00001.parquet",
    "vm_19_1-00000-of-00001.parquet",
    "vm_19_2-00000-of-00001.parquet",
    "vm_1_0-00000-of-00001.parquet",
    "vm_1_1-00000-of-00001.parquet",
    "vm_1_2-00000-of-00001.parquet",
    "vm_20_0-00000-of-00001.parquet",
    "vm_20_1-00000-of-00001.parquet",
    "vm_20_2-00000-of-00001.parquet",
    "vm_21_0-00000-of-00001.parquet",
    "vm_21_1-00000-of-00001.parquet",
    "vm_21_2-00000-of-00001.parquet",
    "vm_22_0-00000-of-00001.parquet",
    "vm_22_1-00000-of-00001.parquet",
    "vm_23_0-00000-of-00001.parquet",
    "vm_23_1-00000-of-00001.parquet",
    "vm_23_2-00000-of-00001.parquet",
    "vm_24_0-00000-of-00001.parquet",
    "vm_24_1-00000-of-00001.parquet",
    "vm_24_2-00000-of-00001.parquet",
    "vm_25_0-00000-of-00001.parquet",
    "vm_25_1-00000-of-00001.parquet",
    "vm_26_0-00000-of-00001.parquet",
    "vm_26_1-00000-of-00001.parquet"
]
 


def download_dataset_parquet_files(dataset_name, split, local_dir, files_to_download=None):
    builder = load_dataset_builder(dataset_name)

    repo_id = '/'.join(dataset_name.split('/')[:2])
    if files_to_download is None:
        files_to_download = [f for f in builder.config.data_files[split] if f.endswith('.parquet')]

    files_info = []
    for file in files_to_download:
        if '::' in file:
            _, file_path = file.split('::', 1)
        else:
            file_path = file
        filename = os.path.basename(file_path)
        files_info.append((filename, file_path))

    os.makedirs(local_dir, exist_ok=True)

    for filename, file_path in files_info:
        if not filename.endswith('.parquet'):
            print(f"Skipping non-parquet file: {filename}")
            continue

        try:
            print(f"Downloading: {filename}")
            local_file = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", local_dir=local_dir)
            print(f"Downloaded: {filename} to {local_file}")
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")

files_to_download_ext = [f"data/{filename}" for filename in files_to_download]
download_dataset_parquet_files(dataset_name, split, local_dir, files_to_download_ext)



def load_and_combine_parquet_files_into_dataset(files):
  local_file_routes = [f"downloaded_parquet_files/data/{file}" for file in files]
  sub_ds_tables = [pq.read_table(route) for route in local_file_routes]
  combined_table = pa.concat_tables(sub_ds_tables)
  dataset = Dataset(combined_table)
  return dataset

my_dataset = load_and_combine_parquet_files_into_dataset(files_to_download)

my_dataset.push_to_hub(dataset_name="amuvarma/1m-fac_0")