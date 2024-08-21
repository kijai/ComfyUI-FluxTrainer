# ComfyUI Flux Trainer

## DISCLAIMER:
I have **very** little previous experience in training anything, Flux is basically first model I've been inspired to learn. Previously I've only trained AnimateDiff Motion Loras, and built similar training nodes for it.

## DO NOT ASK ME FOR TRAINING ADVICE
I can not emphasize this enough, this repository is not for raising questions related to the training itself, that would be better done to kohya's repo. Even so keep in mind my implementation may have mistakes.

# EXPERIMENTAL
Both these nodes and the underlaying implementation by kohya is work in progress and expected to change. 

## Why train in ComfyUI?
- Familiar UI (obviously only if you are a Comfy user already)
- You can use same models you use for inference
- You can use same python environment, I faced no incompabilities
- You can build workflows to compare settings etc.

Currently supports LoRA training, and untested full finetune with code from kohya's scripts: https://github.com/kohya-ss/sd-scripts

![Screenshot 2024-08-21 020207](https://github.com/user-attachments/assets/1686b180-90c8-41d0-8c96-63e76ebc2475)

