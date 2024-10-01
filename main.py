from huggingface_hub import hf_hub_download
import os
from datasets import load_dataset_builder
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset

dataset_name = "CanopyLabs/audio_2m_small"
split = "train"
local_dir = "./downloaded_parquet_files"
files_to_download = [
    "vm_0-00000-of-00001.parquet",
    "vm_107-00000-of-00001.parquet",
    "vm_108-00000-of-00001.parquet",
    "vm_109-00000-of-00001.parquet",
    "vm_11-00000-of-00001.parquet",
    "vm_113-00000-of-00001.parquet",
    "vm_118-00000-of-00001.parquet",
    "vm_119-00000-of-00001.parquet",
    "vm_120-00000-of-00001.parquet",
    "vm_122-00000-of-00001.parquet",
    "vm_124-00000-of-00001.parquet",
    "vm_125-00000-of-00001.parquet",
    "vm_127-00000-of-00001.parquet",
    "vm_128-00000-of-00001.parquet",
    "vm_13-00000-of-00001.parquet",
    "vm_131-00000-of-00001.parquet",
    "vm_135-00000-of-00001.parquet",
    "vm_137-00000-of-00001.parquet",
    "vm_144-00000-of-00001.parquet",
    "vm_146-00000-of-00001.parquet",
    "vm_147-00000-of-00001.parquet",
    "vm_151-00000-of-00001.parquet",
    "vm_152-00000-of-00001.parquet",
    "vm_153-00000-of-00001.parquet",
    "vm_156-00000-of-00001.parquet",
    "vm_157-00000-of-00001.parquet",
    "vm_158-00000-of-00001.parquet",
    "vm_16-00000-of-00001.parquet",
    "vm_163-00000-of-00001.parquet",
    "vm_171-00000-of-00001.parquet",
    "vm_172-00000-of-00001.parquet",
    "vm_173-00000-of-00001.parquet",
    "vm_174-00000-of-00001.parquet",
    "vm_175-00000-of-00001.parquet",
    "vm_179-00000-of-00001.parquet",
    "vm_18-00000-of-00001.parquet",
    "vm_183-00000-of-00001.parquet",
    "vm_184-00000-of-00001.parquet",
    "vm_190-00000-of-00001.parquet",
    "vm_192-00000-of-00001.parquet",
    "vm_194-00000-of-00001.parquet",
    "vm_196-00000-of-00001.parquet",
    "vm_197-00000-of-00001.parquet",
    "vm_2-00000-of-00001.parquet",
    "vm_200-00000-of-00001.parquet",
    "vm_202-00000-of-00001.parquet",
    "vm_214-00000-of-00001.parquet",
    "vm_215-00000-of-00001.parquet",
    "vm_220-00000-of-00001.parquet",
    "vm_221-00000-of-00001.parquet",
    "vm_222-00000-of-00001.parquet",
    "vm_224-00000-of-00001.parquet",
    "vm_227-00000-of-00001.parquet",
    "vm_229-00000-of-00001.parquet",
    "vm_23-00000-of-00001.parquet",
    "vm_231-00000-of-00001.parquet",
    "vm_232-00000-of-00001.parquet",
    "vm_234-00000-of-00001.parquet",
    "vm_235-00000-of-00001.parquet",
    "vm_241-00000-of-00001.parquet",
    "vm_246-00000-of-00001.parquet",
    "vm_248-00000-of-00001.parquet",
    "vm_249-00000-of-00001.parquet",
    "vm_25-00000-of-00001.parquet",
    "vm_250-00000-of-00001.parquet",
    "vm_252-00000-of-00001.parquet",
    "vm_253-00000-of-00001.parquet",
    "vm_254-00000-of-00001.parquet",
    "vm_259-00000-of-00001.parquet",
    "vm_26-00000-of-00001.parquet",
    "vm_262-00000-of-00001.parquet",
    "vm_266-00000-of-00001.parquet",
    "vm_267-00000-of-00001.parquet",
    "vm_27-00000-of-00001.parquet",
    "vm_270-00000-of-00001.parquet",
    "vm_272-00000-of-00001.parquet",
    "vm_275-00000-of-00001.parquet",
    "vm_278-00000-of-00001.parquet",
    "vm_279-00000-of-00001.parquet",
    "vm_281-00000-of-00001.parquet",
    "vm_284-00000-of-00001.parquet",
    "vm_285-00000-of-00001.parquet",
    "vm_289-00000-of-00001.parquet",
    "vm_290-00000-of-00001.parquet",
    "vm_292-00000-of-00001.parquet",
    "vm_293-00000-of-00001.parquet",
    "vm_298-00000-of-00001.parquet",
    "vm_30-00000-of-00001.parquet",
    "vm_304-00000-of-00001.parquet",
    "vm_305-00000-of-00001.parquet",
    "vm_307-00000-of-00001.parquet",
    "vm_31-00000-of-00001.parquet",
    "vm_310-00000-of-00001.parquet",
    "vm_311-00000-of-00001.parquet",
    "vm_315-00000-of-00001.parquet",
    "vm_34-00000-of-00001.parquet",
    "vm_43-00000-of-00001.parquet",
    "vm_45-00000-of-00001.parquet",
    "vm_47-00000-of-00001.parquet",
    "vm_51-00000-of-00001.parquet",
    "vm_52-00000-of-00001.parquet",
    "vm_55-00000-of-00001.parquet",
    "vm_57-00000-of-00001.parquet",
    "vm_60-00000-of-00001.parquet",
    "vm_67-00000-of-00001.parquet",
    "vm_68-00000-of-00001.parquet",
    "vm_69-00000-of-00001.parquet",
    "vm_71-00000-of-00001.parquet",
    "vm_77-00000-of-00001.parquet",
    "vm_78-00000-of-00001.parquet",
    "vm_8-00000-of-00001.parquet",
    "vm_81-00000-of-00001.parquet",
    "vm_88-00000-of-00001.parquet",
    "vm_89-00000-of-00001.parquet",
    "vm_9-00000-of-00001.parquet",
    "vm_90-00000-of-00001.parquet",
    "vm_91-00000-of-00001.parquet",
    "vm_92-00000-of-00001.parquet",
    "vm_93-00000-of-00001.parquet",
    "vm_97-00000-of-00001.parquet"
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

print("Downloaded all files")



def load_and_combine_parquet_files_into_dataset(files):
  local_file_routes = [f"downloaded_parquet_files/data/{file}" for file in files]
  sub_ds_tables = [pq.read_table(route) for route in local_file_routes]
  combined_table = pa.concat_tables(sub_ds_tables)
  dataset = Dataset(combined_table)
  return dataset

my_dataset = load_and_combine_parquet_files_into_dataset(files_to_download)

my_dataset.push_to_hub("amuvarma/tts-10k-part-3")