# Runable Gill

I add inference code for json file and fix some bugs includng requirements incompatibility and out-of-date sd1.5 model(by changing it to `benjamin-paine/stable-diffusion-v1-5`. You need to log in your huggingface before downloading it). 

If CUDA error exisits during inference, you can reduce `batch_size` or `max_text_length` by passing args.

## Enivironment
```
conda create -n gill python=3.10
conda activate gill
pip install -r requirements.txt
```

## How to run

### Prepare your data

Prepare your data in the following format. You can also add your own dataloader in `data_loader.py`.
You can check arg configs to set your data path.
If you use other dataloader, please remember to add your task and dataloader in `inference.py` line 95(maybe).
```
[
    {
        "id": "10008370",
        "image": null,
        "conversations": [
            {
                "from": "human",
                "content": [
                    {
                        "text": "How to Backlight a Mirror?",
                        "image": null
                    },
                    {
                        "text": "Method 1: Choosing a Mirror and LED Strips",
                        "image": null
                    },
                    {
                        "text": "Step 1: Obtain a floating mirror.\nThere will need to be space between the mirror and the wall in order for your lights to shine from behind your mirror. A floating mirror is installed with the right amount of space behind it; you can shop for a floating mirror in department stores or online. Floating mirrors come with the necessary hardware and instructions to hang them away from your wall.",
                        "image": "./images/v4-460px-Backlight-a-Mirror-Step-1.jpg"
                    }
                ]
            },
        ]
    },
    ...
]
```

### Run
```
python inference.py
```
You can also run `inference.sh` for multi GPUs inference. Remeber to change the commands in it.

## Tips

### args config
| args          | instruction                                    | default          |
|---------------|-----------------------------------------|-----------------|
| `--input_dir`  | Path to the input directory               | `../data`       |
| `--output_dir` | Path to the output directory             | `../data_output`|
| `--task`       | Task name                                 | `wikihow`       |
| `--batch_size` | Batch size                               | `5`             |
| `--begin_idx`            | Beginning index                          | `None`          |
| `--end_idx`              | Ending index                             | `None`          |
| `--max_text_length`   | Max text length   | `2000`    |
