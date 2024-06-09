import os
import pickle
import re
from calAig import calAigeval
from tqdm import tqdm
from multiprocessing import Pool, Manager

data_file_path='./task1/project_data'
initial_train_path='./InitialAIG/train'
initial_test_path='./InitialAIG/test'

allevaldic={}
allevaldic_path='./mytask1/allevaldic.pkl'

def process_file(file_name,progress_queue):
    file_path=os.path.join(data_file_path,file_name)
    circuit_name,_=file_name.split('_')
    if os.path.exists(os.path.join(initial_train_path,circuit_name+'.aig')):
        circuit_path=initial_train_path
    else: 
        circuit_path=initial_test_path
    with open(file_path,'rb') as f:
        dic=pickle.load(f)
    for i in range(0,len(dic['input'])):
        if circuit_path is initial_train_path:
            calAigeval(state=dic['input'][i],circuitPath=circuit_path,nextState='./mytask1/train/aig')
        else:
            calAigeval(state=dic['input'][i],circuitPath=circuit_path,nextState='./mytask1/test/aig')
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
