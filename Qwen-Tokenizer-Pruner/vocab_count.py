import os
from tqdm import tqdm
import json
import torch
from utils import make_context
from langdetect import detect as langdetect
from langdetect import DetectorFactory
from tqdm import tqdm
DetectorFactory.seed = 0 # no random

def get_text_list(folder_path):
    query_list = []
    prompt_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('json'):
            file = json.load(open(os.path.join(folder_path, file_name)))
            if 'query' in file:
                query_list.append(file['query'])
            if 'response' in file:
                prompt_list.append(file['response'])
            if 'prompt' in file:
                prompt_list.append(file['prompt'])
    return query_list, prompt_list


def count_freq_based_on_data(data, old_tokens_dict, tokenizer):
    vocab_counts = {token_id: 0 for token_id in old_tokens_dict.values()} 
    for sentence in data:
        tokens = tokenizer.encode(sentence)
        for token in tokens:
            vocab_counts[token] += 1
            # TODO: how to add special tokens? 

    # Add special tokens to vocab_counts TODO: check if this is correct
    # Retrieve special tokens
    pad_token = tokenizer.pad_token
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    additional_special_tokens = tokenizer.additional_special_tokens

    # Add special tokens to vocab_counts
    vocab_counts[pad_token] = 1
    vocab_counts[bos_token] = 1
    vocab_counts[eos_token] = 1
    for token in additional_special_tokens:
        vocab_counts[token] = 1

    return vocab_counts
    


def is_special_token(token):
    return ((token.startswith('<') and token.endswith('>') and len(token) > 2) or
            (token.startswith('[') and token.endswith(']') and len(token) > 2))
    



def update_vocab_count_by_langfilter(support_lang, vocab_counts, old_tokens_dict, count_offset=1):
    # old_tokens_dict is a dict mapping token string -> token id
    for token_str, token_id in tqdm(old_tokens_dict.items()):
        try:
            # Use language detection on the token string and check if it belongs to the supported languages
            if (langdetect(token_str) in support_lang) or is_special_token(token_str):
                vocab_counts[token_id] += count_offset
        except Exception:
            # If language detection fails, we still update the count
            vocab_counts[token_id] += count_offset
    return vocab_counts # key = token_id, value = count


                

def count_recursive(vocab_size, vocab_counts, old_bytes_list):
    recursive_counts = [0 for _ in range(vocab_size)]

    for i in tqdm(range(len(old_bytes_list))):
        token_bytes = old_bytes_list[i]
        t_count = vocab_counts[i]
        b_len = len(token_bytes)
        if t_count > 0 and b_len > 1:
            for j in range(1, b_len):
                for k in range(b_len+1-j):
                    sub_token = token_bytes[k:j+k]
                    if sub_token in old_bytes_list:
                        recursive_counts[old_bytes_list.index(sub_token)] += t_count

    return recursive_counts