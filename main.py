from huggingface_hub import hf_hub_download
import os
from datasets import load_dataset_builder
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset

dataset_name = "eliasfiz/audio_2m_small"
split = "train"
local_dir = "./downloaded_parquet_files"
files_to_download = [
    "vm_10-00000-of-00001.parquet",
    "vm_111-00000-of-00001.parquet",
    "vm_112-00000-of-00001.parquet",
    "vm_114-00000-of-00001.parquet",
    "vm_115-00000-of-00001.parquet",
    "vm_116-00000-of-00001.parquet",
    "vm_117-00000-of-00001.parquet",
    "vm_120-00000-of-00001.parquet",
    "vm_123-00000-of-00001.parquet",
    "vm_126-00000-of-00001.parquet",
    "vm_130-00000-of-00001.parquet",
    "vm_139-00000-of-00001.parquet",
    "vm_141-00000-of-00001.parquet",
    "vm_143-00000-of-00001.parquet",
    "vm_145-00000-of-00001.parquet",
    "vm_148-00000-of-00001.parquet",
    "vm_15-00000-of-00001.parquet",
    "vm_154-00000-of-00001.parquet",
    "vm_155-00000-of-00001.parquet",
    "vm_159-00000-of-00001.parquet",
    "vm_160-00000-of-00001.parquet",
    "vm_161-00000-of-00001.parquet",
    "vm_162-00000-of-00001.parquet",
    "vm_17-00000-of-00001.parquet",
    "vm_176-00000-of-00001.parquet",
    "vm_178-00000-of-00001.parquet",
    "vm_181-00000-of-00001.parquet",
    "vm_182-00000-of-00001.parquet",
    "vm_185-00000-of-00001.parquet",
    "vm_187-00000-of-00001.parquet",
    "vm_19-00000-of-00001.parquet",
    "vm_193-00000-of-00001.parquet",
    "vm_195-00000-of-00001.parquet",
    "vm_20-00000-of-00001.parquet",
    "vm_204-00000-of-00001.parquet",
    "vm_208-00000-of-00001.parquet",
    "vm_21-00000-of-00001.parquet",
    "vm_212-00000-of-00001.parquet",
    "vm_216-00000-of-00001.parquet",
    "vm_217-00000-of-00001.parquet",
    "vm_218-00000-of-00001.parquet",
    "vm_219-00000-of-00001.parquet",
    "vm_22-00000-of-00001.parquet",
    "vm_223-00000-of-00001.parquet",
    "vm_228-00000-of-00001.parquet",
    "vm_236-00000-of-00001.parquet",
    "vm_237-00000-of-00001.parquet",
    "vm_238-00000-of-00001.parquet",
    "vm_24-00000-of-00001.parquet",
    "vm_244-00000-of-00001.parquet",
    "vm_247-00000-of-00001.parquet",
    "vm_251-00000-of-00001.parquet",
    "vm_255-00000-of-00001.parquet",
    "vm_258-00000-of-00001.parquet",
    "vm_260-00000-of-00001.parquet",
    "vm_261-00000-of-00001.parquet",
    "vm_264-00000-of-00001.parquet",
    "vm_265-00000-of-00001.parquet",
    "vm_268-00000-of-00001.parquet",
    "vm_273-00000-of-00001.parquet",
    "vm_276-00000-of-00001.parquet",
    "vm_286-00000-of-00001.parquet",
    "vm_287-00000-of-00001.parquet",
    "vm_291-00000-of-00001.parquet",
    "vm_295-00000-of-00001.parquet",
    "vm_296-00000-of-00001.parquet",
    "vm_297-00000-of-00001.parquet",
    "vm_299-00000-of-00001.parquet",
    "vm_303-00000-of-00001.parquet",
    "vm_306-00000-of-00001.parquet",
    "vm_308-00000-of-00001.parquet",
    "vm_309-00000-of-00001.parquet",
    "vm_312-00000-of-00001.parquet",
    "vm_316-00000-of-00001.parquet",
    "vm_33-00000-of-00001.parquet",
    "vm_34-00000-of-00001.parquet",
    "vm_35-00000-of-00001.parquet",
    "vm_38-00000-of-00001.parquet",
    "vm_4-00000-of-00001.parquet",
    "vm_46-00000-of-00001.parquet",
    "vm_48-00000-of-00001.parquet",
    "vm_49-00000-of-00001.parquet",
    "vm_5-00000-of-00001.parquet",
    "vm_50-00000-of-00001.parquet",
    "vm_53-00000-of-00001.parquet",
    "vm_54-00000-of-00001.parquet",
    "vm_58-00000-of-00001.parquet",
    "vm_59-00000-of-00001.parquet",
    "vm_61-00000-of-00001.parquet",
    "vm_64-00000-of-00001.parquet",
    "vm_66-00000-of-00001.parquet",
    "vm_75-00000-of-00001.parquet",
    "vm_76-00000-of-00001.parquet",
    "vm_79-00000-of-00001.parquet",
    "vm_82-00000-of-00001.parquet",
    "vm_83-00000-of-00001.parquet",
    "vm_84-00000-of-00001.parquet",
    "vm_85-00000-of-00001.parquet",
    "vm_94-00000-of-00001.parquet",
    "vm_95-00000-of-00001.parquet",
    "vm_96-00000-of-00001.parquet",
    "vm_98-00000-of-00001.parquet"
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

my_dataset.push_to_hub("amuvarma/tts-10k-part-1")