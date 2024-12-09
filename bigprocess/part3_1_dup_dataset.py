from datasets import load_dataset
import multiprocessing

num_threads = multiprocessing.cpu_count()

dsn = "eliasfiz/audio_pretrain_10m-facodec"
push_name = "amuvarma/eliasfiz-audio_pretrain_10m-facodec-1dups" 
ds = load_dataset(dsn)
print(ds)


consecutive_count = 0



def remove_excess_consecutive_integers(dataset, column_name, facodec_columns):
    
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
        for idx, col in enumerate(facodec_columns):
            if col in row:
                new_row[col] = [
                    x if isinstance(x, int) else x  
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

processed_dataset = remove_excess_consecutive_integers(dataset, 'facodec_1', ['facodec_1', 'facodec_0', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5'])
# processed_dataset = remove_excess_consecutive_integers(processed_dataset, 'ass1_facodec_2', ['ass2_facodec_1','ass2_facodec_0', 'ass2_facodec_2', 'ass2_facodec_3', 'ass2_facodec_4', 'ass2_facodec_5'])
# processed_dataset = remove_excess_consecutive_integers(processed_dataset, 'ass3_facodec_1', ['ass3_facodec_1','ass3_facodec_0', 'ass3_facodec_2', 'ass3_facodec_3', 'ass3_facodec_4', 'ass3_facodec_5'])

processed_dataset = processed_dataset.remove_columns(['audio'])
processed_dataset.push_to_hub(push_name)