import os
import pickle
import re
from tqdm import tqdm
from multiprocessing import Pool, Manager

from representationAig import representationAig

data_file_path='./mytask1/train/aig'
output_file_path='./mytask1/train/data'

initial_train_path='./InitialAIG/train'
initial_test_path='./InitialAIG/test'

def process_file(file_name,progress_queue):
    file_path=os.path.join(data_file_path,file_name)
    name,_=file_name.split('.')
    output_path=os.path.join(output_file_path,name)
    representationAig(state=file_path,outputPath=output_path)
    progress_queue.put(1)
    
    

def main():
    files = [f for f in os.listdir(data_file_path) if os.path.isfile(os.path.join(data_file_path, f))]
    
    # Create a Manager object to handle the progress queue
    manager = Manager()
    progress_queue = manager.Queue()
    with Pool(processes=16) as pool:
        # Create a progress bar with the total number of files
        progress_bar = tqdm(total=len(files))

        # Define a callback function to update the progress bar
        def update_progress(_):
            progress_bar.update()

        # Process files in parallel and update progress
        for file in files:
            pool.apply_async(process_file, args=(file, progress_queue), callback=update_progress)

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()
        progress_bar.close()

if __name__=="__main__":
    main()