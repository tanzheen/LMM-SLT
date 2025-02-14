import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from training.params import DataArguments
from constants import *
from utils import get_model_name_from_path, load_pretrained_model
import numpy as np
import cv2
import torchvision.transforms as transforms
from vidaug import augmentors as va
import gzip
import pickle

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixel": min_pixel,
                "max_pixel": max_pixel

            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]



def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

# Example augmentation function (random crop during training, center crop for eval)
def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    if is_train:
        left = np.random.randint(0, resize[0] - crop_size)
        top = np.random.randint(0, resize[1] - crop_size)
    else:
        left = (resize[0] - crop_size) // 2
        top = (resize[1] - crop_size) // 2
    return (left, top, left + crop_size, top + crop_size), resize

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

TRANSLATION_PROMPT="Translate this sign language for me into German"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = load_dataset_file(data_path)
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.raw_data = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps
        self.list = [key for key, value in self.raw_data.items()]
        
        # Video augmentation setup
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            sometimes(va.RandomTranslate(x=10, y=10)),
        ])

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.raw_data[i]
        # each iteration of the list: 
            # ['image']
            # ['video']
            # how to prepare the frames of the video 
        

        processor = self.processor

        
        if "video" in sources:
            is_dummy = False
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            def is_video_file(filename):
                video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
                return any(filename.lower().endswith(ext) for ext in video_extensions)
            if not is_video_file(video_files):
                raise ValueError("THIS IS NOT A VIDEO")

            tgt_sample = sources["text"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.data_args.fps)
                videos.append(video_input)
        else:
            raise ValueError("No video or image found in the dataset")

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"

            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}user\n{VISION_START_TOKEN}{DEFAULT_VIDEO_TOKEN}{VISION_END_TOKEN}Can you translate this sign language for me into German?\n{DEFAULT_IM_END_TOKEN}\n"
            gpt_response = f"{DEFAULT_IM_START_TOKEN}assistant\n{translation}{DEFAULT_IM_END_TOKEN}"


            if DEFAULT_VIDEO_TOKEN in user_input:
                inputs = processor(text=[user_input],  videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                all_second_gird.extend(inputs["second_per_grid_ts"])
            else: raise ValueError("Video not processed properly")
            prompt_input_ids = inputs['input_ids']
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])


     
            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)


        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        assert input_ids.shape == labels.shape

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            is_dummy=is_dummy,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict





class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_dummy_flags = []
        batch_second_per_grid_ts = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_dummy_flags.append(example["is_dummy"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'is_dummy': torch.tensor(batch_dummy_flags, dtype=torch.bool)
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

def replace_image_tokens(input_string, is_video=False):

    if is_video:
        input_string = input_string.replace(LLAVA_VIDEO_TOKEN+'\n', VISION_START_TOKEN+DEFAULT_VIDEO_TOKEN+VISION_END_TOKEN)

    else:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)

    return input_string

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)