# ComfyUI Flux Trainer

Wrapper for slightly modified kohya's training scripts: https://github.com/kohya-ss/sd-scripts

Including code from: https://github.com/KohakuBlueleaf/Lycoris

And https://github.com/LoganBooker/prodigy-plus-schedule-free

## DISCLAIMER:
I have **very** little previous experience in training anything, Flux is basically first model I've been inspired to learn. Previously I've only trained AnimateDiff Motion Loras, and built similar training nodes for it.

## DO NOT ASK ME FOR TRAINING ADVICE
I can not emphasize this enough, this repository is not for raising questions related to the training itself, that would be better done to kohya's repo. Even so keep in mind my implementation may have mistakes.

The default settings aren't necessarily any good, they are just the last (out of many) I've tried and worked for my dataset.

# THIS IS EXPERIMENTAL
Both these nodes and the underlaying implementation by kohya is work in progress and expected to change. 

# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-FluxTrainer\requirements.txt`

In addition torch version 2.4.0 or higher is highly recommended.

Example workflow for LoRA training can be found in the examples folder, it utilizes additional nodes from:

https://github.com/kijai/ComfyUI-KJNodes

And some (optional) debugging nodes from:

https://github.com/rgthree/rgthree-comfy

For LoRA training the models need to be the normal fp8 or fp16 versions, also make sure the VAE is the non-diffusers version:

https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors

For full model training the fp16 version of the main model needs to be used.

## Why train in ComfyUI?
- Familiar UI (obviously only if you are a Comfy user already)
- You can use same models you use for inference
- You can use same python environment, I faced no incompabilities
- You can build workflows to compare settings etc.

Currently supports LoRA training, and untested full finetune with code from kohya's scripts: https://github.com/kohya-ss/sd-scripts

Experimental support for LyCORIS training has been added as well, using code from: https://github.com/KohakuBlueleaf/Lycoris

![Screenshot 2024-08-21 020207](https://github.com/user-attachments/assets/1686b180-90c8-41d0-8c96-63e76ebc2475)

