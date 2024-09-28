from datasets import load_dataset
import multiprocessing

num_threads = multiprocessing.cpu_count()

ds_name = "amuvarma/1m-fac_0"
ds = load_dataset(ds_name)



def remove_excess_consecutive_integers(dataset):
    def process_row(row):
        facodec_1 = row['facodec_1']
        result = []
        indices_to_keep = []
        consecutive_count = 1
        last_num = None
        
        for i, num in enumerate(facodec_1):
            if isinstance(num, int):
                if num == last_num:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                
                if consecutive_count <= 3:
                    result.append(num)
                    indices_to_keep.append(i)
                
                last_num = num
            else:
                result.append(num)
                indices_to_keep.append(i)
                consecutive_count = 1
                last_num = None
        
        # Update facodec columns
        new_row = row.copy()
        facodec_columns = ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']
        for col in facodec_columns:
            if col in row:
                new_row[col] = [row[col][i] for i in indices_to_keep if i < len(row[col])]
        
        return new_row

    # Use map to process each row
    processed_dataset = dataset.map(
        process_row, 
        num_proc=num_threads,
        )
    
    return processed_dataset

# Use the function
dataset = ds['train']  # Assuming you want to process the 'train' split
processed_dataset = remove_excess_consecutive_integers(dataset)


processed_dataset.push_to_hub("amuvarma/750k-raw_dups3-0")