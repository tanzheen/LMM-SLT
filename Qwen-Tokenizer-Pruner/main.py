import os
import json
import torch
import argparse
import base64
import transformers
print(f"Transformers version: {transformers.__version__}")
print(f"Available Qwen classes: {[cls for cls in dir(transformers) if 'Qwen' in cls]}")
from transformers import AutoProcessor, AutoModelForImageTextToText
from vocab_count import count_freq, update_vocab_count_by_langfilter, count_recursive
from vocab_save import get_new_vocab_and_map, save_vocab, save_vocab_hf, reduce_to_target_size
from model_save import *
from utils import get_bpe_file
from tqdm import tqdm

from langdetect import detect as langdetect
from langdetect import DetectorFactory
import gzip
import pickle

# 确保检测结果的一致性
DetectorFactory.seed = 0


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def main():
    # start vocabulary pruning
    print('============ Start Qwen Vocabulary Pruning ==========')

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model_path', type=str, default="unsloth/Qwen2.5-3B-4bit")
    parser.add_argument('--new_model_path', type=str, default="unsloth/Qwen2.5-3B-4bit-pruned-P14")
    parser.add_argument('--support_data', type=str, default="data/Phonexi-2014T/labels.train")
    parser.add_argument('--support_lang', default=[], type=str, nargs='+')
    parser.add_argument('--inherit_vocab_count', type=str, default=None)
    parser.add_argument('--target_size', type=int, default=None)
    args = parser.parse_args()
    
    # valid check
    assert (args.support_data is not None) or (len(args.support_lang) > 0), "Must provide at least one pruning method." 

    # init output path
    if not os.path.exists(args.new_model_path):
        os.makedirs(args.new_model_path)
        print(f"==> Create output folder: {args.new_model_path}")
    
    # load old model and tokenizer
    print(f"==> Load old model and tokenizer from: {args.old_model_path}")
    old_model = AutoModelForImageTextToText.from_pretrained(
    args.old_model_path) 
    old_tokenizer = AutoProcessor.from_pretrained(args.old_model_path)
    old_vocab_size = old_model.config.__dict__['vocab_size'] 
    print(f"Tokenizer has vocabulary size {old_vocab_size}") 
    # Extract vocabulary (tokens are stored as keys in the tokenizer's vocab)
    old_tokens_dict = old_tokenizer.tokenizer.get_vocab() # dictionary of 
    # using support data
    if args.support_data is not None:
        print(f"==> Loading Support Data (for Freqs Count) from: {args.support_data}")
        raw_data = load_dataset_file('data/Phonexi-2014T/labels.train')
        data = [] 
        for key, value in raw_data.items():
            sentence = value['text']
            data.append(sentence)

        vocab_counts = count_freq_based_on_data(data=data, 
                                  old_tokens_dict=old_tokens_dict, 
                                  tokenizer=old_tokenizer.tokenizer)
        
    else: 

        vocab_counts = {token_id: 0 for token_id in old_tokens_dict.values()} 
    # # using supported language to filter
    if len(args.support_lang) > 0:
        print(f"==> Using support language to filter old vocabulary")
        print(f"Supported Language: {args.support_lang}")
        vocab_counts = update_vocab_count_by_langfilter(support_lang=args.support_lang, 
                                                        vocab_counts=vocab_counts, 
                                                        old_tokens_dict=old_tokens_dict, 
                                                        count_offset=1)
        #print(f"new vocab_counts: {vocab_counts}")
    # save vocab_counts
        
    # TODO: i do not know what this is for
    # # #sub-token count
    # # print(f"==> Recursively calculate sub-token count")
    # # recur_counts = count_recursive(vocab_size=old_vocab_size, 
    # #                                vocab_counts=vocab_counts, 
    # #                                )

    
    # get new vocab
    print(f"==> Get new vocabulary bpe file and save it") 
    new_tokens_dict, mapping_new2old = get_new_vocab_and_map(old_tokens_dict=old_tokens_dict,
                                                            old_vocab_size=old_vocab_size,
                                                            vocab_counts=vocab_counts)
    
    
    # # save the new tokenizer
    # save_vocab_hf(new_tokens_dict, mapping_new2old, args.new_model_path, old_tokenizer)
    # save_vocab_hf(new_bytes_list, mapping_new2old, args.new_model_path)
    mapping_new2old = list(torch.load(os.path.join(args.new_model_path, 'token_mapping.torch')))
    
    # update model ckpt
    new_vocab_size = len(mapping_new2old)
    print(f"==> Detected as Qwen-VL model")
    saving_updated_qwenvl(old_model, new_vocab_size, mapping_new2old, args.new_model_path)
    old_tokenizer.save_pretrained(args.new_model_path)
if __name__=='__main__':
    main()