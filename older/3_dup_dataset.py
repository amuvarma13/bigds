from datasets import load_dataset
import multiprocessing

num_threads = multiprocessing.cpu_count()

dsn = "eliasfiz/audio"
push_name = "amuvarma/dev-fac-raw" 
ds = load_dataset(dsn, split='train')
print(ds)


consecutive_count = 0

ds = ds.select(range(100))


def remove_excess_consecutive_integers(dataset, column_name):
    
    def process_row(row):
        if row[column_name] is None:
            return row
        
        facodec_1 = row[column_name]
        result = []
        indices_to_keep = []

        last_num = None
        
        for i, num in enumerate(facodec_1):
            if isinstance(num, int):
                if num == last_num:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                
                if consecutive_count <= 1:
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
                new_row[col] = [
    x + 128266 + (i*1024) if isinstance(x, int) else x 
    for x in (row[col][i] for i in indices_to_keep if i < len(row[col]))
]


        
        return new_row

    # Use map to process each row
    processed_dataset = dataset.map(
        process_row, 
        num_proc=num_threads,
        )
    
    return processed_dataset

# Use the function
dataset = ds  # Assuming you want to process the 'train' split

processed_dataset = remove_excess_consecutive_integers(dataset, 'facodec_1')


processed_dataset_60k = processed_dataset.select(range(60000))
processed_dataset_60k.push_to_hub("amuvarma/60k-fac-with-audio-1dups")

processed_dataset_40k = processed_dataset.select(range(60000, 100000))
processed_dataset_40k.push_to_hub("amuvarma/40k-fac-with-audio-1dups")


processed_dataset.push_to_hub(push_name)