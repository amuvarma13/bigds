from huggingface_hub import hf_hub_download
import os
from datasets import load_dataset_builder
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset

dataset_name = "CanopyElias/audio_2m_small"
split = "train"
local_dir = "./downloaded_parquet_files"
files_to_download = [
    "vm_1-00000-of-00001.parquet",
    "vm_100-00000-of-00001.parquet",
    "vm_101-00000-of-00001.parquet",
    "vm_102-00000-of-00001.parquet",
    "vm_103-00000-of-00001.parquet",
    "vm_104-00000-of-00001.parquet",
    "vm_105-00000-of-00001.parquet",
    "vm_106-00000-of-00001.parquet",
    "vm_110-00000-of-00001.parquet",
    "vm_113-00000-of-00001.parquet",
    "vm_12-00000-of-00001.parquet",
    "vm_121-00000-of-00001.parquet",
    "vm_122-00000-of-00001.parquet",
    "vm_124-00000-of-00001.parquet",
    "vm_129-00000-of-00001.parquet",
    "vm_132-00000-of-00001.parquet",
    "vm_133-00000-of-00001.parquet",
    "vm_134-00000-of-00001.parquet",
    "vm_136-00000-of-00001.parquet",
    "vm_138-00000-of-00001.parquet",
    "vm_14-00000-of-00001.parquet",
    "vm_140-00000-of-00001.parquet",
    "vm_142-00000-of-00001.parquet",
    "vm_149-00000-of-00001.parquet",
    "vm_150-00000-of-00001.parquet",
    "vm_164-00000-of-00001.parquet",
    "vm_165-00000-of-00001.parquet",
    "vm_166-00000-of-00001.parquet",
    "vm_167-00000-of-00001.parquet",
    "vm_168-00000-of-00001.parquet",
    "vm_169-00000-of-00001.parquet",
    "vm_170-00000-of-00001.parquet",
    "vm_177-00000-of-00001.parquet",
    "vm_180-00000-of-00001.parquet",
    "vm_186-00000-of-00001.parquet",
    "vm_188-00000-of-00001.parquet",
    "vm_189-00000-of-00001.parquet",
    "vm_191-00000-of-00001.parquet",
    "vm_198-00000-of-00001.parquet",
    "vm_199-00000-of-00001.parquet",
    "vm_201-00000-of-00001.parquet",
    "vm_203-00000-of-00001.parquet",
    "vm_205-00000-of-00001.parquet",
    "vm_206-00000-of-00001.parquet",
    "vm_207-00000-of-00001.parquet",
    "vm_209-00000-of-00001.parquet",
    "vm_210-00000-of-00001.parquet",
    "vm_211-00000-of-00001.parquet",
    "vm_213-00000-of-00001.parquet",
    "vm_223-00000-of-00001.parquet",
    "vm_225-00000-of-00001.parquet",
    "vm_226-00000-of-00001.parquet",
    "vm_230-00000-of-00001.parquet",
    "vm_233-00000-of-00001.parquet",
    "vm_239-00000-of-00001.parquet",
    "vm_240-00000-of-00001.parquet",
    "vm_242-00000-of-00001.parquet",
    "vm_243-00000-of-00001.parquet",
    "vm_245-00000-of-00001.parquet",
    "vm_256-00000-of-00001.parquet",
    "vm_257-00000-of-00001.parquet",
    "vm_263-00000-of-00001.parquet",
    "vm_269-00000-of-00001.parquet",
    "vm_271-00000-of-00001.parquet",
    "vm_274-00000-of-00001.parquet",
    "vm_277-00000-of-00001.parquet",
    "vm_28-00000-of-00001.parquet",
    "vm_280-00000-of-00001.parquet",
    "vm_282-00000-of-00001.parquet",
    "vm_283-00000-of-00001.parquet",
    "vm_288-00000-of-00001.parquet",
    "vm_29-00000-of-00001.parquet",
    "vm_294-00000-of-00001.parquet",
    "vm_3-00000-of-00001.parquet",
    "vm_300-00000-of-00001.parquet",
    "vm_301-00000-of-00001.parquet",
    "vm_302-00000-of-00001.parquet",
    "vm_313-00000-of-00001.parquet",
    "vm_314-00000-of-00001.parquet",
    "vm_32-00000-of-00001.parquet",
    "vm_36-00000-of-00001.parquet",
    "vm_37-00000-of-00001.parquet",
    "vm_39-00000-of-00001.parquet",
    "vm_40-00000-of-00001.parquet",
    "vm_41-00000-of-00001.parquet",
    "vm_42-00000-of-00001.parquet",
    "vm_44-00000-of-00001.parquet",
    "vm_47-00000-of-00001.parquet",
    "vm_56-00000-of-00001.parquet",
    "vm_6-00000-of-00001.parquet",
    "vm_62-00000-of-00001.parquet",
    "vm_63-00000-of-00001.parquet",
    "vm_65-00000-of-00001.parquet",
    "vm_7-00000-of-00001.parquet",
    "vm_70-00000-of-00001.parquet",
    "vm_72-00000-of-00001.parquet",
    "vm_73-00000-of-00001.parquet",
    "vm_74-00000-of-00001.parquet",
    "vm_80-00000-of-00001.parquet",
    "vm_86-00000-of-00001.parquet",
    "vm_87-00000-of-00001.parquet",
    "vm_95-00000-of-00001.parquet",
    "vm_99-00000-of-00001.parquet"
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

my_dataset.push_to_hub("amuvarma/tts-10k-part-2")