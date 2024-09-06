import os
from gill import utils
import json

def WikihowDataloader(input_path, begin_idx, end_idx):
    with open(os.path.join(input_path, 'data.json'), 'r') as f:
        data = json.load(f)

    if end_idx is not None:
        data = data[:end_idx]
    if begin_idx is not None:
        data = data[begin_idx:]

    prompts = {}
    for d in data:
        id = d['id']
        content = d['conversations'][0]['content']
        prompts[id] = []
        for c in content:
            prompts[id].append(c['text'] + '\n')
            if c['image'] is not None:
                img_path = os.path.join(input_path, c['image'])
                if os.path.exists(img_path):
                    img = utils.get_image_from_path(img_path),
                    prompts[id].append(img)
        # prompts[id].append("Q: What should I do next? Explain it to me in detail.\nA:")

    return prompts