import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
import json
import torch

from gill import models
from gill import utils
from data_loader import WikihowDataloader

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on Gill")
    parser.add_argument('--input_dir', type=str, default='../data', 
                        help="Path to the input directory (default: '../data')")
    parser.add_argument('--output_dir', type=str, default='../data_output', 
                        help="Path to the output directory (default: '../data_output')")
    parser.add_argument('--task', type=str, default='wikihow', 
                        help="Task name (default: 'wikihow')")
    parser.add_argument('--batch_size', type=int, default=5, 
                        help="Batch size (default: 5)")
    parser.add_argument('--begin_idx', type=int, default=None, 
                        help="Beginning index (default: None)")
    parser.add_argument('--end_idx', type=int, default=None, 
                        help="Ending index (default: None)")
    parser.add_argument('--max_text_length', type=int, default=2000, 
                        help="Max text length (default: 2000)")
                        
    args = parser.parse_args()
    return args

def generate_dialogue(prompts: list, system_message: str = None, num_words: int = 32,
                      sf: float = 1.0, temperature: float = 0.0, top_p: float = 1.0,
                      divider_count: int = 40):
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)

    full_outputs = {}
    if system_message:
        print("Adding system message")
        full_inputs = [system_message]
    else:
        full_inputs = []

    answers = {}
    images = {}
    for id, prompt in prompts.items():
        formatted_prompt = []
        for p in prompt:
            if type(p) == Image.Image:
                full_inputs.append(p)
                formatted_prompt.append(p)
            elif type(p) == str:
                full_inputs.append(p)
                formatted_prompt.append(f'User: {p}')
        formatted_prompt.append('=' * divider_count)  # Add divider

        try:
            return_outputs = model.generate_for_images_and_texts(
                full_inputs, num_words=num_words, ret_scale_factor=sf,
                generator=g_cuda, temperature=temperature, top_p=top_p)

            # Add outputs
            output_text = return_outputs[0].replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', '')
            full_inputs.append(output_text + '\n')

            formatted_return_outputs = []
            answers[id] = ""
            images[id] = []
            for p in return_outputs:
                if type(p) == str:
                    p_formatted = p.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', '')
                    p_formatted = p_formatted.replace('[IMG0][IMG1][IMG2][IMG3][IMG4][IMG5][IMG6][IMG7]', '')
                    answers[id] += p_formatted
                    formatted_return_outputs.append(f'GILL: {p_formatted}')
                else:
                    formatted_return_outputs.append(p)
                    images[id].append(p)
            formatted_return_outputs.append('=' * divider_count)  # Add divider

            full_outputs[id] = formatted_prompt + formatted_return_outputs
        except Exception as e:
            print(e)

    return full_outputs, answers, images

if __name__ == "__main__":
    args = parse_args()
    print(args)

    input_path = os.path.join(args.input_dir,args.task)
    output_path = os.path.join(args.output_dir,args.task)
    output_temp_path = os.path.join(output_path, 'temp')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_temp_path, exist_ok=True)

    # add your own task here
    if 'wikihow' in args.task:
        prompts = WikihowDataloader(input_path, args.begin_idx, args.end_idx, args.max_text_length)
    
    prompts_items = list(prompts.items())
    prompts_batch = [dict(prompts_items[i:i + args.batch_size]) for i in range(0, len(prompts_items), args.batch_size)]

    # Download the model checkpoint and embeddings to checkpoints/gill_opt/
    model_dir = 'checkpoints/gill_opt/'
    model = models.load_gill(model_dir)

    sf = 1.4  # Scaling factor: increase to increase the chance of returning an image
    temperature = 0.6  # 0 means deterministic, try 0.6 for more randomness
    top_p = 0.95  # If you set temperature to 0.6, set this to 0.95
    num_words = 50

    output_data = []
    for prompts in tqdm(prompts_batch, total=len(prompts_batch)):
        _, answers, images = generate_dialogue(prompts, num_words=num_words, sf=sf, temperature=temperature, top_p=top_p)
        for id, image_set in images.items():
            img_paths = []
            os.makedirs(f'{output_path}/images/{id}',exist_ok=True)
            for i,image in enumerate(image_set):
                if image['decision'][0] == 'gen':
                    img = image['gen'][0][0].resize((512, 512))
                    img_path = f'images/{id}/{i}_gen.png'
                else:
                    img = image['ret'][0][0].resize((512, 512))
                    img_path = f'images/{id}/{i}_ret.png'
                img.save(f'{output_path}/{img_path}')
                img_paths.append(img_path)
            d = {}
            d['id'] = id
            d['answer'] = answers[id]
            d['images'] = img_paths
            output_data.append(d)
        with open(os.path.join(output_temp_path,f'data_{args.begin_idx}_{args.end_idx}_temp.json'),'w') as f:
            json.dump(output_data, f, indent=4)

    with open(os.path.join(output_path,f'data_{args.begin_idx}_{args.end_idx}.json'),'w') as f:
        json.dump(output_data, f, indent=4)
