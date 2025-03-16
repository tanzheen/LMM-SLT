import os
import shutil
import torch
import torch
import os

def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path):
    # Ensure we access the correct embedding layer
    embedding_layer = old_model.model.embed_tokens  # Correctly referencing embedding layer

    # Define new embedding and LM head layers
    new_embeds = torch.nn.Embedding(new_vocab_size, old_model.config.hidden_size, dtype=embedding_layer.weight.dtype)
    print(f"new_embeds: {new_embeds}")
    new_lm_head = torch.nn.Linear(old_model.config.hidden_size, new_vocab_size, bias=False, dtype=old_model.lm_head.weight.dtype)

    # Convert token mapping to tensor and move to the correct device
    token_mapping_tensor = torch.LongTensor(token_mapping).to("cpu")

    # Copy old weights into new embeddings and LM head
    new_embeds.weight.data = embedding_layer.weight.data[token_mapping_tensor]
    new_lm_head.weight.data = old_model.lm_head.weight.data[token_mapping_tensor]

    # Update model with new layers
    old_model.model.embed_tokens = new_embeds
    old_model.lm_head = new_lm_head
    old_model.config.vocab_size = new_vocab_size

    # Update visual token mapping if exists
    if "visual" in old_model.config.__dict__:
        old_model.config.visual["image_start_id"] = token_mapping.index(old_model.config.visual["image_start_id"])

    # Helper function to update token ids (handles list or single integer)
    def update_token_id(token_id):
        if isinstance(token_id, list):
            return [token_mapping.index(t) for t in token_id]
        else:
            return token_mapping.index(token_id)
    # update config
    old_model.config.eos_token_id = update_token_id(old_model.config.eos_token_id)
    old_model.config.pad_token_id = update_token_id(old_model.config.pad_token_id)
    old_model.config.image_token_id = update_token_id(old_model.config.image_token_id)

    # Update generation config for eos_token_id and pad_token_id
    old_model.generation_config.eos_token_id = update_token_id(old_model.generation_config.eos_token_id)
    old_model.generation_config.pad_token_id = update_token_id(old_model.generation_config.pad_token_id)
    old_model.generation_config.bos_token_id = update_token_id(old_model.generation_config.bos_token_id)
    # Save the updated model
    print(f"Saving new model checkpoint to {output_path}")
    old_model.save_pretrained(output_path)
    print(f"old_model: {old_model}")





def saving_updated_qwen(old_model, new_vocab_size, token_mapping, output_path):
    # define new module
    new_embeds = torch.nn.Embedding(new_vocab_size, old_model.config.hidden_size, dtype=old_model.transformer.wte.weight.dtype)
    new_lm_head = torch.nn.Linear(old_model.config.hidden_size, new_vocab_size, bias=False, dtype=old_model.lm_head.weight.dtype)
    # get new module parameter from the old
    assert len(set(token_mapping)) == new_vocab_size
    new_embeds.weight.data = old_model.transformer.wte.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    new_lm_head.weight.data = old_model.lm_head.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    # update model
    old_model.transformer.wte.weight = new_embeds.weight
    old_model.lm_head.weight = new_lm_head.weight
    old_model.transformer.wte.num_embeddings = new_vocab_size
    old_model.lm_head.out_features = new_vocab_size
    # update config
    old_model.config.__dict__['vocab_size'] = new_vocab_size
    old_model.config.__dict__['_name_or_path'] = output_path
    old_model.generation_config.__dict__['eos_token_id'] = token_mapping.index(old_model.generation_config.__dict__['eos_token_id'])
    old_model.generation_config.__dict__['pad_token_id'] = token_mapping.index(old_model.generation_config.__dict__['pad_token_id'])
    # save new model
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path)