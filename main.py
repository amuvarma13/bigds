from huggingface_hub import hf_hub_download
import os
from datasets import load_dataset_builder
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset

dataset_name = "eliasfiz/audio_10m"
split = "train"
local_dir = "./downloaded_parquet_files"
files_to_download = [
    "vm_100-00000-of-00001.parquet",
    "vm_101-00000-of-00001.parquet",
    "vm_103-00000-of-00001.parquet",
    "vm_107-00000-of-00001.parquet",
    "vm_111-00000-of-00001.parquet",
    "vm_112-00000-of-00001.parquet", 
    "vm_113-00000-of-00001.parquet",
    "vm_115-00000-of-00001.parquet",
    "vm_121-00000-of-00001.parquet",
    "vm_126-00000-of-00001.parquet",
    "vm_128-00000-of-00001.parquet",
    "vm_13-00000-of-00001.parquet",
    "vm_130-00000-of-00001.parquet",
    "vm_132-00000-of-00001.parquet",
    "vm_137-00000-of-00001.parquet",
    "vm_138-00000-of-00001.parquet",
    "vm_144-00000-of-00001.parquet",
    "vm_148-00000-of-00001.parquet",
    "vm_150-00000-of-00001.parquet",
    "vm_151-00000-of-00001.parquet",
    "vm_152-00000-of-00001.parquet",
    "vm_153-00000-of-00001.parquet",
    "vm_158-00000-of-00001.parquet",
    "vm_159-00000-of-00001.parquet",
    "vm_16-00000-of-00001.parquet",
    "vm_161-00000-of-00001.parquet",
    "vm_163-00000-of-00001.parquet",
    "vm_169-00000-of-00001.parquet",
    "vm_17-00000-of-00001.parquet",
    "vm_175-00000-of-00001.parquet",
    "vm_176-00000-of-00001.parquet",
    "vm_177-00000-of-00001.parquet",
    "vm_178-00000-of-00001.parquet",
    "vm_18-00000-of-00001.parquet",
    "vm_182-00000-of-00001.parquet",
    "vm_184-00000-of-00001.parquet",
    "vm_185-00000-of-00001.parquet",
    "vm_187-00000-of-00001.parquet",
    "vm_19-00000-of-00001.parquet",
    "vm_193-00000-of-00001.parquet",
    "vm_194-00000-of-00001.parquet",
    "vm_195-00000-of-00001.parquet",
    "vm_197-00000-of-00001.parquet",
    "vm_2-00000-of-00001.parquet",
    "vm_202-00000-of-00001.parquet",
    "vm_204-00000-of-00001.parquet",
    "vm_209-00000-of-00001.parquet",
    "vm_22-00000-of-00001.parquet",
    "vm_222-00000-of-00001.parquet",
    "vm_233-00000-of-00001.parquet",
    "vm_241-00000-of-00001.parquet",
    "vm_244-00000-of-00001.parquet",
    "vm_25-00000-of-00001.parquet",
    "vm_250-00000-of-00001.parquet",
    "vm_251-00000-of-00001.parquet",
    "vm_262-00000-of-00001.parquet",
    "vm_266-00000-of-00001.parquet",
    "vm_268-00000-of-00001.parquet",
    "vm_27-00000-of-00001.parquet",
    "vm_273-00000-of-00001.parquet",
    "vm_290-00000-of-00001.parquet",
    "vm_292-00000-of-00001.parquet",
    "vm_296-00000-of-00001.parquet",
    "vm_3-00000-of-00001.parquet",
    "vm_300-00000-of-00001.parquet",
    "vm_329-00000-of-00001.parquet",
    "vm_33-00000-of-00001.parquet",
    "vm_332-00000-of-00001.parquet",
    "vm_333-00000-of-00001.parquet",
    "vm_334-00000-of-00001.parquet",
    "vm_335-00000-of-00001.parquet",
    "vm_336-00000-of-00001.parquet",
    "vm_338-00000-of-00001.parquet",
    "vm_35-00000-of-00001.parquet",
    "vm_38-00000-of-00001.parquet",
    "vm_45-00000-of-00001.parquet",
    "vm_50-00000-of-00001.parquet",
    "vm_52-00000-of-00001.parquet",
    "vm_55-00000-of-00001.parquet",
    "vm_6-00000-of-00001.parquet",
    "vm_61-00000-of-00001.parquet",
    "vm_62-00000-of-00001.parquet",
    "vm_63-00000-of-00001.parquet",
    "vm_67-00000-of-00001.parquet",
    "vm_7-00000-of-00001.parquet",
    "vm_74-00000-of-00001.parquet",
    "vm_77-00000-of-00001.parquet",
    "vm_78-00000-of-00001.parquet",
    "vm_80-00000-of-00001.parquet",
    "vm_82-00000-of-00001.parquet",
    "vm_83-00000-of-00001.parquet",
    "vm_87-00000-of-00001.parquet",
    "vm_88-00000-of-00001.parquet",
    "vm_89-00000-of-00001.parquet",
    "vm_9-00000-of-00001.parquet",
    "vm_90-00000-of-00001.parquet",
    "vm_92-00000-of-00001.parquet",
    "vm_94-00000-of-00001.parquet"
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

my_dataset.push_to_hub("amuvarma/6-interleave-800k-0")