from datasets import load_dataset
import datasets

ds_name = "amuvarma/1m-fac_0"
dsy = load_dataset(ds_name)


ds = dsy["train"]



def remove_consecutive_duplicates(ds):
    def process_row(row):
        facodec_1 = row['facodec_1']
        indices_to_keep = [0] + [i for i in range(1, len(facodec_1)) if facodec_1[i] != facodec_1[i-1]]

        for col in ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']:
            row[col] = [row[col][i] for i in indices_to_keep]

        return row

    return ds.map(process_row)

# Usage
ds = remove_consecutive_duplicates(ds)


def add_offset_to_facodec(ds, offset=256003):
    def process_row(row):
        for col in ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']:
            row[col] = [x + offset for x in row[col]]
        row["facodec_1_shifted"] = row["facodec_1"]
        return row

    return ds.map(process_row)

# Usage
ds = add_offset_to_facodec(ds)

max_length = 2000

def prepare_dataset_for_model(ds):
    # First, find the maximum length across all rows


    def process_row(row):
        # Rename facodec_0 to input_ids
        row['input_ids'] = row['facodec_0']

        # Duplicate facodec_1, pop the first element, shift, and append 256002
        row['facodec_1_shifted'] = row['facodec_1_shifted'][1:] + [256002]

        # Create attention_mask before padding
        original_length = len(row['input_ids'])
        row['attention_mask'] = [1] * original_length + [0] * (max_length - original_length)

        # Pad input_ids
        row['input_ids'] = row['input_ids'] + [0] * (max_length - original_length)

        # Pad facodec_1
        row['facodec_1'] = row['facodec_1'] + [0] * (max_length - len(row['facodec_1']))

        row['facodec_1_shifted'] = row['facodec_1_shifted'] + [0] * (max_length - len(row['facodec_1_shifted']))

        return row

    # Apply the processing to each row
    ds = ds.map(process_row)

    # Set the format of the dataset to convert lists to pytorch tensors
    ds = ds.with_format("torch")

    return ds

# Usage
ds = prepare_dataset_for_model(ds)

ds.push_to_hub("amuvarma/multilayer-1m-0")