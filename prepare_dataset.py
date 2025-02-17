import gzip
import pickle

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
# i just need to keep the text , and the name which can be modified to be the video path 
for dataset in ["train", "dev", "test"]:
    dataset = load_dataset_file(f"data/Phonexi-2014T/labels.{dataset}")
    for key in dataset:
        print(dataset[key]['text'])





