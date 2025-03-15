import os
import torch
import base64
from tqdm import tqdm
import json 
from transformers import PreTrainedTokenizerFast

def reduce_to_target_size(old_vocab_size, target_vocab_size, vocab_counts, recur_counts, old_bytes_list):
    total_count_with_idx = [(vocab_counts[i] + recur_counts[i], i) for i in range(old_vocab_size)]
    sorted_count_with_idx = sorted(total_count_with_idx, key=lambda x: x[0])
    remove_count = 0
    remove_target = old_vocab_size - target_vocab_size

    for i in tqdm(range(len(sorted_count_with_idx))):
        token_count, token_idx = sorted_count_with_idx[i]
        if remove_count >= remove_target:
            continue
        elif token_count == 0:
            remove_count += 1
        elif len(old_bytes_list[token_idx]) > 1:
            # whether it can be represented by sub-token
            token = old_bytes_list[token_idx]
            b_len = len(token)
            for j in range(1, b_len):
                if (token[:j] in old_bytes_list) and (token[j:] in old_bytes_list):
                    parta_index = old_bytes_list.index(token[:j])
                    partb_index = old_bytes_list.index(token[j:])
                    if (vocab_counts[parta_index] + recur_counts[parta_index] > 0) and (vocab_counts[partb_index] + recur_counts[partb_index] > 0):
                        vocab_counts[token_idx] = 0
                        recur_counts[token_idx] = 0
                        remove_count += 1
                        break
                    
    if remove_count < remove_target:
        print(f"Failed to reach the target size")
                    
    return vocab_counts, recur_counts


def get_new_vocab_and_map(old_tokens_dict, old_vocab_size, vocab_counts):
    """
    Create a new vocabulary dictionary and a mapping from new token indices to the original token IDs.
    
    Args:
        old_tokens_dict (dict): Original vocabulary mapping, where keys are token strings and values are old token IDs.
        old_vocab_size (int): The original vocabulary size.
        vocab_counts (list): A list of frequency counts where vocab_counts[old_id] gives the frequency of the token with that ID.
        
    Returns:
        new_tokens_dict (dict): New vocabulary mapping token string -> new contiguous token ID.
        mapping_new2old (list): A list where mapping_new2old[new_id] = old token ID for the token kept.
    """
    new_tokens_dict = {}   # New vocab: token string -> new token id (0, 1, 2, ...)
    mapping_new2old = []   # Mapping from new token id to old token id

    # Iterate over tokens sorted by their original token IDs to preserve the original order.
    for token_str, old_id in sorted(old_tokens_dict.items(), key=lambda x: x[1]):
        # If the token's frequency is greater than 0, keep it.
        if vocab_counts[old_id] > 0:
            # New token ID is the current length of new_tokens_dict (which is contiguous)
            new_id = len(new_tokens_dict)
            new_tokens_dict[token_str] = new_id
            mapping_new2old.append(old_id) # index in the list would be the new id and the value would be the old id

    print(f"Vocabulary size: {old_vocab_size} => New vocab size: {len(new_tokens_dict)}")
    return new_tokens_dict, mapping_new2old 
# new_tokens_dict: token string -> new token id (0, 1, 2, ...), mapping_new2old: new token id -> old token id


def save_vocab(bytes_list, token_mapping, output_path):
    new_tiktoken_path = os.path.join(output_path, 'qwen.tiktoken')
    token_mapping_path = os.path.join(output_path, 'token_mapping.torch')
    # saving new tiktoken_bpe_file
    with open(new_tiktoken_path, "w", encoding="utf8") as w:
        for i, token in enumerate(bytes_list):
            line = base64.b64encode(token).decode("utf8") + " " + str(i) + "\n"
            w.write(line)
    print(f"New Tiktoken BPE file (size: {len(bytes_list)}) is saved to {new_tiktoken_path}")

    # saving mapping index
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
    print(f"Mapping file (new token 2 old token) is saved: {token_mapping_path}")



def save_vocab_hf(new_tokens_dict, token_mapping, output_path, old_tokenizer):
    vocab_file = os.path.join(output_path, 'vocab.json')
    merges_file = os.path.join(output_path, 'merges.txt')
    token_mapping_path = os.path.join(output_path, 'token_mapping.torch')

    # Creating vocabulary dictionary by decoding bytes to strings
    vocab_dict = {
        token_str: token_id 
        for token_str, token_id in new_tokens_dict.items()
    }


    # Saving vocab.json (Hugging Face style)
    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f)
    print(f"Hugging Face vocab file is saved to {vocab_file}")

    # Saving merges.txt (needed for BPE models, but can be left empty if unnecessary)
    with open(merges_file, "w") as f:
        f.write("")
    print(f"Hugging Face merges file is saved to {merges_file}")

    # Saving token mapping
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
    print(f"Mapping file (new token to old token) is saved to {token_mapping_path}")

   
