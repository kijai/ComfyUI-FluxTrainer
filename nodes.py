import os
import torch
from torchvision import transforms

import folder_paths
import comfy.model_management as mm
import comfy.utils
import toml
import json
import time
import shutil
from pathlib import Path
script_directory = os.path.dirname(os.path.abspath(__file__))

from .flux_train_network_comfy import FluxNetworkTrainer
from .library import flux_train_utils as  flux_train_utils
from .flux_train_comfy import FluxTrainer
from .flux_train_comfy import setup_parser as train_setup_parser
from .library.device_utils import init_ipex
init_ipex()

from .library import train_util
from .train_network import setup_parser as train_network_setup_parser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxTrainModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "transformer": (folder_paths.get_filename_list("unet"), ),
            "vae": (folder_paths.get_filename_list("vae"), ),
            "clip_l": (folder_paths.get_filename_list("clip"), ),
            "t5": (folder_paths.get_filename_list("clip"), ),
           },
        }

    RETURN_TYPES = ("TRAIN_FLUX_MODELS",)
    RETURN_NAMES = ("flux_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer"

    def loadmodel(self, transformer, vae, clip_l, t5):
        
        transformer_path = folder_paths.get_full_path("unet", transformer)
        vae_path = folder_paths.get_full_path("vae", vae)
        clip_path = folder_paths.get_full_path("clip", clip_l)
        t5_path = folder_paths.get_full_path("clip", t5)

        flux_models = {
            "transformer": transformer_path,
            "vae": vae_path,
            "clip_l": clip_path,
            "t5": t5_path
        }
        
        return (flux_models,)

class TrainDatasetGeneralConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "color_aug": ("BOOLEAN",{"default": False, "tooltip": "enable weak color augmentation"}),
            "flip_aug": ("BOOLEAN",{"default": False, "tooltip": "enable horizontal flip augmentation"}),
            "shuffle_caption": ("BOOLEAN",{"default": False, "tooltip": "shuffle caption"}),
            "caption_dropout_rate": ("FLOAT",{"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "tag dropout rate"}),
            "alpha_mask": ("BOOLEAN",{"default": False, "tooltip": "use alpha channel as mask for training"}),
            },
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("dataset_general",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, shuffle_caption, caption_dropout_rate, color_aug, flip_aug, alpha_mask):
        
        dataset = {
           "general": {
                "shuffle_caption": shuffle_caption,
                "caption_extension": ".txt",
                "keep_tokens_separator": "|||",
                "caption_dropout_rate": caption_dropout_rate,
                "color_aug": color_aug,
                "flip_aug": flip_aug,
           },
           "datasets": []
        }
        dataset_json = json.dumps(dataset, indent=2)
        #print(dataset_json)
        dataset_config = {
            "datasets": dataset_json,
            "alpha_mask": alpha_mask
        }
        return (dataset_config,)

class TrainDatasetAdd:
    def __init__(self):
        self.previous_dataset_signature = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "dataset_config": ("JSON",),
            "width": ("INT",{"min": 64, "default": 1024, "tooltip": "base resolution width"}),
            "height": ("INT",{"min": 64, "default": 1024, "tooltip": "base resolution height"}),
            "batch_size": ("INT",{"min": 1, "default": 2, "tooltip": "Higher batch size uses more memory and generalizes the training more"}),
            "dataset_path": ("STRING",{"multiline": True, "default": "", "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "class_tokens": ("STRING",{"multiline": True, "default": "", "tooltip": "aka trigger word, if specified, will be added to the start of each caption, if no captions exist, will be used on it's own"}),
            "enable_bucket": ("BOOLEAN",{"default": True, "tooltip": "enable buckets for multi aspect ratio training"}),
            "bucket_no_upscale": ("BOOLEAN",{"default": False, "tooltip": "don't allow upscaling when bucketing"}),
            "num_repeats": ("INT", {"default": 1, "min": 1, "tooltip": "number of times to repeat dataset for an epoch"}),
            "min_bucket_reso": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 8, "tooltip": "min bucket resolution"}),
            "max_bucket_reso": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "max bucket resolution"}),
  
            },
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, dataset_config, dataset_path, class_tokens, width, height, batch_size, num_repeats, enable_bucket,  
                  bucket_no_upscale, min_bucket_reso, max_bucket_reso):
        
        new_dataset = {
            "resolution": (width, height),
            "batch_size": batch_size,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale,
            "min_bucket_reso": min_bucket_reso,
            "max_bucket_reso": max_bucket_reso,
            "subsets": [
                {
                    "image_dir": dataset_path,
                    "class_tokens": class_tokens,
                    "num_repeats": num_repeats
                }
            ]
        }

        # Generate a signature for the new dataset
        new_dataset_signature = self.generate_signature(new_dataset)

        # Load the existing datasets
        existing_datasets = json.loads(dataset_config["datasets"])

        # Remove the previously added dataset if it exists
        if self.previous_dataset_signature:
            existing_datasets["datasets"] = [
                ds for ds in existing_datasets["datasets"]
                if self.generate_signature(ds) != self.previous_dataset_signature
            ]

        # Add the new dataset
        existing_datasets["datasets"].append(new_dataset)

        # Store the new dataset signature for future runs
        self.previous_dataset_signature = new_dataset_signature

        # Convert back to JSON and update dataset_config
        updated_dataset_json = json.dumps(existing_datasets, indent=2)
        dataset_config["datasets"] = updated_dataset_json

        return dataset_config,

    def generate_signature(self, dataset):
        # Create a unique signature for the dataset based on its attributes
        return json.dumps(dataset, sort_keys=True)

class OptimizerConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optimizer_type": (["adamw8bit", "adamw","prodigy", "CAME", "Lion8bit", "Lion"], {"default": "adamw8bit", "tooltip": "optimizer type"}),
            "max_grad_norm": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"], {"default": "constant", "tooltip": "learning rate scheduler"}),
            "lr_warmup_steps": ("INT",{"default": 0, "min": 0, "tooltip": "learning rate warmup steps"}),
            "lr_scheduler_num_cycles": ("INT",{"default": 1, "min": 1, "tooltip": "learning rate scheduler num cycles"}),
            "lr_scheduler_power": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "learning rate scheduler power"}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, min_snr_gamma, extra_optimizer_args, **kwargs):
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        kwargs["optimizer_args"] = [arg.strip() for arg in extra_optimizer_args.strip().split(',') if arg.strip()]
        return (kwargs,)

class OptimizerConfigAdafactor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "max_grad_norm": ("FLOAT",{"default": 0.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "adafactor"], {"default": "constant_with_warmup", "tooltip": "learning rate scheduler"}),
            "lr_warmup_steps": ("INT",{"default": 0, "min": 0, "tooltip": "learning rate warmup steps"}),
            "lr_scheduler_num_cycles": ("INT",{"default": 1, "min": 1, "tooltip": "learning rate scheduler num cycles"}),
            "lr_scheduler_power": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "learning rate scheduler power"}),
            "relative_step": ("BOOLEAN",{"default": False, "tooltip": "relative step"}),
            "scale_parameter": ("BOOLEAN",{"default": False, "tooltip": "scale parameter"}),
            "warmup_init": ("BOOLEAN",{"default": False, "tooltip": "warmup init"}),
            "clip_threshold": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "clip threshold"}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, relative_step, scale_parameter, warmup_init, clip_threshold, min_snr_gamma, extra_optimizer_args, **kwargs):
        kwargs["optimizer_type"] = "adafactor"
        extra_args = [arg.strip() for arg in extra_optimizer_args.strip().split(',') if arg.strip()]
        node_args = [
                f"relative_step={relative_step}",
                f"scale_parameter={scale_parameter}",
                f"warmup_init={warmup_init}",
                f"clip_threshold={clip_threshold}"
            ]
        kwargs["optimizer_args"] = node_args + extra_args
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        
        return (kwargs,)

class OptimizerConfigProdigy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "max_grad_norm": ("FLOAT",{"default": 0.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "adafactor"], {"default": "constant", "tooltip": "learning rate scheduler"}),
            "lr_warmup_steps": ("INT",{"default": 0, "min": 0, "tooltip": "learning rate warmup steps"}),
            "lr_scheduler_num_cycles": ("INT",{"default": 1, "min": 1, "tooltip": "learning rate scheduler num cycles"}),
            "lr_scheduler_power": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "learning rate scheduler power"}),
            "weight_decay": ("FLOAT",{"default": 0.0, "tooltip": "weight decay (L2 penalty)"}),
            "decouple": ("BOOLEAN",{"default": True, "tooltip": "use AdamW style weight decay"}),
            "use_bias_correction": ("BOOLEAN",{"default": False, "tooltip": "turn on Adam's bias correction"}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, weight_decay, decouple, min_snr_gamma, use_bias_correction, extra_optimizer_args, **kwargs):
        kwargs["optimizer_type"] = "prodigy"
        extra_args = [arg.strip() for arg in extra_optimizer_args.strip().split(',') if arg.strip()]
        node_args = [
                f"weight_decay={weight_decay}",
                f"decouple={decouple}",
                f"use_bias_correction={use_bias_correction}"
            ]
        kwargs["optimizer_args"] = node_args + extra_args
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        
        return (kwargs,)    

class InitFluxLoRATraining:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset": ("JSON",),
            "optimizer_settings": ("ARGS",),
            "output_name": ("STRING", {"default": "flux_lora", "multiline": False}),
            "output_dir": ("STRING", {"default": "flux_trainer_output", "multiline": False, "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "network_dim": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1, "tooltip": "network dim"}),
            "network_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 256.0, "step": 0.01, "tooltip": "network alpha"}),
            "learning_rate": ("FLOAT", {"default": 4e-4, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"}),
            #"unet_lr": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.00001, "tooltip": "unet learning rate"}),
            #"max_train_epochs": ("INT", {"default": 4, "min": 1, "max": 1000, "step": 1, "tooltip": "max number of training epochs"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"}),
            #"text_encoder_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.00001, "tooltip": "text encoder learning rate"}),
            "apply_t5_attn_mask": ("BOOLEAN", {"default": True, "tooltip": "apply t5 attention mask"}),
            #"t5xxl_max_token_length": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "dev uses 512, schnell 256"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "split_mode": ("BOOLEAN", {"default": False, "tooltip": "[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"}),
            "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"],),
            "logit_mean": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean to use when using the logit_normal weighting scheme"}),
            "logit_std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "std to use when using the logit_normal weighting scheme"}),
            "mode_scale": ("FLOAT", {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scale of mode weighting scheme. Only effective when using the mode as the weighting_scheme"}),
            "timestep_sampling": (["sigmoid", "uniform", "sigma", "shift", "flux_shift"], {"tooltip": "Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal and shift of sigmoid (recommend value of 3.1582 for discrete_flow_shift)"}),
            "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for sigmoid timestep sampling (only used when timestep-sampling is sigmoid"}),
            "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"tooltip": "How to interpret and process the model prediction: raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0, "step": 0.01, "tooltip": "guidance scale, for Flux training should be 1.0"}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "for the Euler Discrete Scheduler, default is 3.0"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "fp8_base": ("BOOLEAN", {"default": True, "tooltip": "use fp8 for base model"}),
            "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "fp32", "tooltip": "the actual dtype training uses"}),
            "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn"], {"default": "bf16", "tooltip": "the dtype to save checkpoints as"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "sdpa", "tooltip": "memory efficient attention mode"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "additional args to pass to the training command"}),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume args to pass to the training command"}),
                "train_clip_l": (['disabled', 'use_gradient_dtype', 'use_fp8'], {"default": 'disabled', "tooltip": "also train the clip_l text encoder using specified dtype"}),
                "text_encoder_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.00001, "tooltip": "text encoder learning rate"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS",)
    RETURN_NAMES = ("network_trainer", "epochs_count", "args",)
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, dataset, optimizer_settings, sample_prompts, output_name, attention_mode, 
                      gradient_dtype, save_dtype, split_mode, additional_args=None, resume_args=None, train_clip_l='disabled', **kwargs,):
        mm.soft_empty_cache()
        
        output_dir = os.path.abspath(kwargs.get("output_dir"))
        os.makedirs(output_dir, exist_ok=True)
    
        total, used, free = shutil.disk_usage(output_dir)
 
        required_free_space = 2 * (2**30)
        if free <= required_free_space:
            raise ValueError(f"Insufficient disk space. Required: {required_free_space/2**30}GB. Available: {free/2**30}GB")
        
        dataset_config = dataset["datasets"]
        dataset_toml = toml.dumps(json.loads(dataset_config))

        parser = train_network_setup_parser()
        if additional_args is not None:
            args, _ = parser.parse_known_args(args=[additional_args])
        else:
            args, _ = parser.parse_known_args()
        #print(args)

        if kwargs.get("cache_latents") == "memory":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = False
        elif kwargs.get("cache_latents") == "disk":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = True
            kwargs["caption_dropout_rate"] = 0.0
            kwargs["shuffle_caption"] = False
            kwargs["token_warmup_step"] = 0.0
            kwargs["caption_tag_dropout_rate"] = 0.0
        else:
            kwargs["cache_latents"] = False
            kwargs["cache_latents_to_disk"] = False

        if kwargs.get("cache_text_encoder_outputs") == "memory":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = False
        elif kwargs.get("cache_text_encoder_outputs") == "disk":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = True
        else:
            kwargs["cache_text_encoder_outputs"] = False
            kwargs["cache_text_encoder_outputs_to_disk"] = False

        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        config_dict = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": flux_models["transformer"],
            "clip_l": flux_models["clip_l"],
            "t5xxl": flux_models["t5"],
            "ae": flux_models["vae"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "gradient_checkpointing": True,
            "network_module": ".networks.lora_flux",
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{save_dtype}",
            "loss_type": "l2",
            "text_encoder_lr": 0,
            "t5xxl_max_token_length": 512,
            "alpha_mask": dataset["alpha_mask"],
            "network_train_unet_only": True if train_clip_l == 'disabled' else False,
            "fp8_base_unet": True if train_clip_l=='use_gradient_dtype' else False,
        }
        attention_settings = {
            "sdpa": {"mem_eff_attn": True, "xformers": False, "spda": True},
            "xformers": {"mem_eff_attn": True, "xformers": True, "spda": False}
        }
        config_dict.update(attention_settings.get(attention_mode, {}))

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"}
        }
        config_dict.update(gradient_dtype_settings.get(gradient_dtype, {}))

        split_mode_settings = {
            True: {"split_mode": True, "network_args": ["train_blocks=single"]},
            False: {"split_mode": False, "network_args": ["train_blocks=all"]}
        }
        config_dict.update(split_mode_settings.get(split_mode, {}))

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)

        with torch.inference_mode(False):
            network_trainer = FluxNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)

class InitFluxTraining:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset": ("JSON",),
            "optimizer_settings": ("ARGS",),
            "output_name": ("STRING", {"default": "flux", "multiline": False}),
            "output_dir": ("STRING", {"default": "flux_trainer_output", "multiline": False, "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "learning_rate": ("FLOAT", {"default": 4e-6, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"}),
            "apply_t5_attn_mask": ("BOOLEAN", {"default": True, "tooltip": "apply t5 attention mask"}),
            "t5xxl_max_token_length": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "dev uses 512, schnell 256"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"],),
            "logit_mean": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean to use when using the logit_normal weighting scheme"}),
            "logit_std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "std to use when using the logit_normal weighting scheme"}),
            "mode_scale": ("FLOAT", {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scale of mode weighting scheme. Only effective when using the mode as the weighting_scheme"}),
            "loss_type": (["l1", "l2", "huber", "smooth_l1"], {"default": "l2", "tooltip": "loss type"}),
            "timestep_sampling": (["sigmoid", "uniform", "sigma", "shift", "flux_shift"], {"tooltip": "Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal and shift of sigmoid (recommend value of 3.1582 for discrete_flow_shift)"}),
            "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for sigmoid timestep sampling (only used when timestep-sampling is sigmoid"}),
            "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"tooltip": "How to interpret and process the model prediction: raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)"}),
            "cpu_offload_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "offload the gradient checkpointing to CPU. This reduces VRAM usage for about 2GB"}),
            "optimizer_fusing": (['fused_backward_pass', 'blockwise_fused_optimizers'], {"tooltip": "reduces memory use"}),
            "single_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "number of single blocks to swap. The default is 0. This option must be combined with blockwise_fused_optimizers"}),
            "double_blocks_to_swap": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1, "tooltip": "number of double blocks to swap. This option must be combined with blockwise_fused_optimizers"}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0, "step": 0.01, "tooltip": "guidance scale"}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "for the Euler Discrete Scheduler, default is 3.0"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "fp8_base": ("BOOLEAN", {"default": False, "tooltip": "use fp8 for base model"}),
            "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "bf16", "tooltip": "to use the full fp16/bf16 training"}),
            "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn"], {"default": "bf16", "tooltip": "the dtype to save checkpoints as"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "sdpa", "tooltip": "memory efficient attention mode"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "additional args to pass to the training command"}),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume args to pass to the training command"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS")
    RETURN_NAMES = ("network_trainer", "epochs_count", "args")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, optimizer_settings, dataset, sample_prompts, output_name, 
                      attention_mode, gradient_dtype, save_dtype, optimizer_fusing, additional_args=None, resume_args=None, **kwargs,):
        mm.soft_empty_cache()

        output_dir = os.path.abspath(kwargs.get("output_dir"))
        os.makedirs(output_dir, exist_ok=True)
    
        total, used, free = shutil.disk_usage(output_dir)
        required_free_space = 25 * (2**30)
        if free <= required_free_space:
            raise ValueError(f"Most likely insufficient disk space to complete training. Required: {required_free_space/2**30}GB. Available: {free/2**30}GB")

        dataset_toml = toml.dumps(json.loads(dataset))
        
        parser = train_setup_parser()
        if additional_args is not None:
            args, _ = parser.parse_known_args(args=[additional_args])
        else:
            args, _ = parser.parse_known_args()

        if kwargs.get("cache_latents") == "memory":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = False
        elif kwargs.get("cache_latents") == "disk":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = True
            kwargs["caption_dropout_rate"] = 0.0
            kwargs["shuffle_caption"] = False
            kwargs["token_warmup_step"] = 0.0
            kwargs["caption_tag_dropout_rate"] = 0.0
        else:
            kwargs["cache_latents"] = False
            kwargs["cache_latents_to_disk"] = False

        if kwargs.get("cache_text_encoder_outputs") == "memory":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = False
        elif kwargs.get("cache_text_encoder_outputs") == "disk":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = True
        else:
            kwargs["cache_text_encoder_outputs"] = False
            kwargs["cache_text_encoder_outputs_to_disk"] = False

        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        config_dict = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": flux_models["transformer"],
            "clip_l": flux_models["clip_l"],
            "t5xxl": flux_models["t5"],
            "ae": flux_models["vae"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "gradient_checkpointing": True,
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_{save_dtype}",
            "mem_eff_save": True,

        }
        optimizer_fusing_settings = {
            "fused_backward_pass": {"fused_backward_pass": True, "blockwise_fused_optimizers": False},
            "blockwise_fused_optimizers": {"fused_backward_pass": False, "blockwise_fused_optimizers": True}
        }
        config_dict.update(optimizer_fusing_settings.get(optimizer_fusing, {}))

        attention_settings = {
            "sdpa": {"mem_eff_attn": True, "xformers": False, "spda": True},
            "xformers": {"mem_eff_attn": True, "xformers": True, "spda": False}
        }
        config_dict.update(attention_settings.get(attention_mode, {}))

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"}
        }
        config_dict.update(gradient_dtype_settings.get(gradient_dtype, {}))

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)

        with torch.inference_mode(False):
            network_trainer = FluxTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        
        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)

class InitFluxTrainingFromPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset_settings": ("TOML_DATASET",),
            "preset_args": ("KOHYA_ARGS",),
            "output_name": ("STRING", {"default": "flux", "multiline": False}),
            "output_dir": ("STRING", {"default": "flux_trainer_output", "multiline": False, "tooltip": "output directory, root is ComfyUI folder"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "STRING", "KOHYA_ARGS")
    RETURN_NAMES = ("network_trainer", "epochs_count", "output_path", "args")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, dataset_settings, sample_prompts, output_name, preset_args, **kwargs,):
        mm.soft_empty_cache()

        dataset = dataset_settings["dataset"]
        dataset_repeats = dataset_settings["repeats"]
        
        parser = train_setup_parser()
        args, _ = parser.parse_known_args()
        for key, value in vars(preset_args).items():
            setattr(args, key, value)
        
        output_dir = os.path.join(script_directory, "output")
        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        width, height = toml.loads(dataset)["datasets"][0]["resolution"]
        config_dict = {
            "sample_prompts": prompts,
            "dataset_repeats": dataset_repeats,
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": flux_models["transformer"],
            "clip_l": flux_models["clip_l"],
            "t5xxl": flux_models["t5"],
            "ae": flux_models["vae"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "gradient_checkpointing": True,
            "dataset_config": dataset,
            "output_dir": output_dir,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{args.save_precision}",
            "width" : int(width),
            "height" : int(height),

        }

        config_dict.update(kwargs)

        for key, value in config_dict.items():
            setattr(args, key, value)

        with torch.inference_mode(False):
            network_trainer = FluxNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        final_output_path = os.path.join(output_dir, output_name)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, final_output_path, args)
    
class FluxTrainLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1, "tooltip": "the step point in training to validate/save"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT",)
    RETURN_NAMES = ("network_trainer", "steps",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer"

    def train(self, network_trainer, steps):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]
            initial_global_step = network_trainer.global_step

            target_global_step = network_trainer.global_step + steps
            pbar = comfy.utils.ProgressBar(steps)
            while network_trainer.global_step < target_global_step:
                steps_done = training_loop(
                    break_at_steps = target_global_step,
                    epoch = network_trainer.current_epoch.value,
                )
                pbar.update(steps_done)
               
                # Also break if the global steps have reached the max train steps
                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break

            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, network_trainer.global_step)

class FluxTrainSave:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": False, "tooltip": "save the whole model state as well"}),
            "copy_to_comfy_lora_folder": ("BOOLEAN", {"default": False, "tooltip": "copy the lora model to the comfy lora folder"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT",)
    RETURN_NAMES = ("network_trainer","lora_path", "steps",)
    FUNCTION = "save"
    CATEGORY = "FluxTrainer"

    def save(self, network_trainer, save_state, copy_to_comfy_lora_folder):
        import shutil
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step
            
            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            trainer.save_model(ckpt_name, trainer.accelerator.unwrap_model(trainer.network), global_step, trainer.current_epoch.value + 1)

            remove_step_no = train_util.get_remove_step_no(trainer.args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, remove_step_no)
                trainer.remove_model(remove_ckpt_name)

            if save_state:
                train_util.save_and_remove_state_stepwise(trainer.args, trainer.accelerator, global_step)

            lora_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_lora_folder:
                destination_dir = os.path.join(folder_paths.models_dir, "loras", "flux_trainer")
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(lora_path, os.path.join(destination_dir, ckpt_name))
        
            
        return (network_trainer, lora_path, global_step)

class FluxTrainSaveModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "copy_to_comfy_model_folder": ("BOOLEAN", {"default": False, "tooltip": "copy the lora model to the comfy lora folder"}),
            "end_training": ("BOOLEAN", {"default": False, "tooltip": "end the training"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT",)
    RETURN_NAMES = ("network_trainer","model_path", "steps",)
    FUNCTION = "save"
    CATEGORY = "FluxTrainer"

    def save(self, network_trainer, copy_to_comfy_model_folder, end_training):
        import shutil
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step
            
            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            flux_train_utils.save_flux_model_on_epoch_end_or_stepwise(
                trainer.args, 
                False,
                trainer.accelerator,
                trainer.save_dtype,
                trainer.current_epoch.value,
                trainer.num_train_epochs,
                global_step,
                trainer.accelerator.unwrap_model(trainer.unet)
                )

            model_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_model_folder:
                shutil.copy(model_path, os.path.join(folder_paths.models_dir, "diffusion_models", "flux_trainer", ckpt_name))
                model_path = os.path.join(folder_paths.models_dir, "diffusion_models", "flux_trainer", ckpt_name)
            if end_training:
                trainer.accelerator.end_training()
        
        return (network_trainer, model_path, global_step)
    
class FluxTrainEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": True}),
             },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("lora_name", "metadata", "lora_path",)
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer"

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]
            
            network_trainer.metadata["ss_epoch"] = str(network_trainer.num_train_epochs)
            network_trainer.metadata["ss_training_finished_at"] = str(time.time())

            network = network_trainer.accelerator.unwrap_model(network_trainer.network)

            network_trainer.accelerator.end_training()

            if save_state:
                train_util.save_state_on_train_end(network_trainer.args, network_trainer.accelerator)

            ckpt_name = train_util.get_last_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as)
            network_trainer.save_model(ckpt_name, network, network_trainer.global_step, network_trainer.num_train_epochs, force_sync_upload=True)
            logger.info("model saved.")

            final_lora_name = str(network_trainer.args.output_name)
            final_lora_path = os.path.join(network_trainer.args.output_dir, ckpt_name)

            # metadata
            metadata = json.dumps(network_trainer.metadata, indent=2)

            training_loop = None
            network_trainer = None
            mm.soft_empty_cache()
            
        return (final_lora_name, metadata, final_lora_path)

class FluxTrainResume:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "load_state_path": ("STRING", {"default": "", "multiline": True, "tooltip": "path to load state from"}),
            "skip_until_initial_step" : ("BOOLEAN", {"default": False}),
             },
        }

    RETURN_TYPES = ("ARGS", )
    RETURN_NAMES = ("resume_args", )
    FUNCTION = "resume"
    CATEGORY = "FluxTrainer"

    def resume(self, load_state_path, skip_until_initial_step):
        resume_args ={
            "resume": load_state_path,
            "skip_until_initial_step": skip_until_initial_step
        }
            
        return (resume_args, )
    
class FluxTrainValidationSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 20, "min": 1, "max": 256, "step": 1, "tooltip": "sampling steps"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image width"}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image height"}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "guidance scale"}),
            "seed": ("INT", {"default": 42,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "shift": ("BOOLEAN", {"default": True, "tooltip": "shift the schedule to favor high timesteps for higher signal images"}),
            "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
            "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("VALSETTINGS", )
    RETURN_NAMES = ("validation_settings", )
    FUNCTION = "set"
    CATEGORY = "FluxTrainer"

    def set(self, **kwargs):
        validation_settings = kwargs
        print(validation_settings)

        return (validation_settings,)
        
class FluxTrainValidate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
            },
            "optional": {
                "validation_settings": ("VALSETTINGS",),
            }
        }

    RETURN_TYPES = ("NETWORKTRAINER", "IMAGE",)
    RETURN_NAMES = ("network_trainer", "validation_images",)
    FUNCTION = "validate"
    CATEGORY = "FluxTrainer"

    def validate(self, network_trainer, validation_settings=None):
        training_loop = network_trainer["training_loop"]
        network_trainer = network_trainer["network_trainer"]

        params = (
            network_trainer.accelerator, 
            network_trainer.args, 
            network_trainer.current_epoch.value, 
            network_trainer.global_step,
            network_trainer.unet,
            network_trainer.vae,
            network_trainer.text_encoder,
            network_trainer.sample_prompts_te_outputs,
            validation_settings
        )

        split_mode = getattr(network_trainer.args, 'split_mode', False)
        if split_mode:
            image_tensors = network_trainer.sample_images_split_mode(*params)
        else:
            image_tensors = flux_train_utils.sample_images(*params)

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, (0.5 * (image_tensors + 1.0)).cpu().float(),)
    
class VisualizeLoss:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "plot_style": (plt.style.available,{"default": 'default', "tooltip": "matplotlib plot style"}),
            "window_size": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1, "tooltip": "the window size of the moving average"}),
            "normalize_y": ("BOOLEAN", {"default": True, "tooltip": "normalize the y-axis to 0"}),
            "width": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 2, "tooltip": "width of the plot in pixels"}),
            "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 2, "tooltip": "height of the plot in pixels"}),
            "log_scale": ("BOOLEAN", {"default": False, "tooltip": "use log scale on the y-axis"}),
             },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("plot", "loss_list",)
    FUNCTION = "draw"
    CATEGORY = "FluxTrainer"

    def draw(self, network_trainer, window_size, plot_style, normalize_y, width, height, log_scale):
        import numpy as np
        loss_values = network_trainer["network_trainer"].loss_recorder.global_loss_list

        # Apply moving average
        def moving_average(values, window_size):
            return np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        if window_size > 0:
            loss_values = moving_average(loss_values, window_size)

        plt.style.use(plot_style)

        # Convert pixels to inches (assuming 100 pixels per inch)
        width_inches = width / 100
        height_inches = height / 100

        # Create a plot
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.plot(loss_values, label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        if normalize_y:
            plt.ylim(bottom=0)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image = Image.open(buf).convert('RGB')

        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()

        return image_tensor, loss_values,

class FluxKohyaInferenceSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
            "lora_method": (["apply", "merge"], {"tooltip": "whether to apply or merge the lora weights"}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 256, "step": 1, "tooltip": "sampling steps"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image width"}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image height"}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "guidance scale"}),
            "seed": ("INT", {"default": 42,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "use_fp8": ("BOOLEAN", {"default": True, "tooltip": "use fp8 weights"}),
            "prompt": ("STRING", {"multiline": True, "default": "illustration of a kitten", "tooltip": "prompt"}),
          
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "sample"
    CATEGORY = "FluxTrainer"

    def sample(self, flux_models, lora_name, steps, width, height, guidance_scale, seed, prompt, use_fp8, lora_method):

        from .library import flux_utils as flux_utils
        from .library import strategy_flux as strategy_flux
        from .networks import lora_flux as lora_flux
        from typing import List, Optional, Callable
        from tqdm import tqdm
        import einops
        import math
        import accelerate
        import gc

        device = "cuda"
        apply_t5_attn_mask = True

        if use_fp8:
            accelerator = accelerate.Accelerator(mixed_precision="bf16")
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.float16
            accelerator = None
        loading_device = "cpu"
        ae_dtype = torch.bfloat16

        pretrained_model_name_or_path = flux_models["transformer"]
        clip_l = flux_models["clip_l"]
        t5xxl = flux_models["t5"]
        ae = flux_models["vae"]
        lora_path = folder_paths.get_full_path("loras", lora_name)

        # load clip_l
        logger.info(f"Loading clip_l from {clip_l}...")
        clip_l = flux_utils.load_clip_l(clip_l, None, loading_device)
        clip_l.eval()

        logger.info(f"Loading t5xxl from {t5xxl}...")
        t5xxl = flux_utils.load_t5xxl(t5xxl, None, loading_device)
        t5xxl.eval()

        if use_fp8:
            clip_l = accelerator.prepare(clip_l)
            t5xxl = accelerator.prepare(t5xxl)

        t5xxl_max_length = 512
        tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
        encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

        # DiT
        model = flux_utils.load_flow_model("dev", pretrained_model_name_or_path, dtype, loading_device)
        model.eval()
        logger.info(f"Casting model to {dtype}")
        model.to(dtype)  # make sure model is dtype
        if use_fp8:
            model = accelerator.prepare(model)

        # AE
        ae = flux_utils.load_ae("dev", ae, ae_dtype, loading_device)
        ae.eval()
        #if is_fp8(ae_dtype):
        #    ae = accelerator.prepare(ae)

        # LoRA
        lora_models: List[lora_flux.LoRANetwork] = []
        multiplier = 1.0

        lora_model, weights_sd = lora_flux.create_network_from_weights(
            multiplier, lora_path, ae, [clip_l, t5xxl], model, None, True
        )
        if lora_method == "merge":
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        elif lora_method == "apply":
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {lora_name}: {info}")
            lora_model.eval()
            lora_model.to(device)
        lora_models.append(lora_model)


        packed_latent_height, packed_latent_width = math.ceil(height / 16), math.ceil(width / 16)
        noise = torch.randn(
            1,
            packed_latent_height * packed_latent_width,
            16 * 2 * 2,
            device=device,
            dtype=ae_dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

        # prepare embeddings
        logger.info("Encoding prompts...")
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        clip_l = clip_l.to(device)
        t5xxl = t5xxl.to(device)
        with torch.no_grad():
            if use_fp8:
                clip_l.to(ae_dtype)
                t5xxl.to(ae_dtype)
                with accelerator.autocast():
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )
        # NaN check
        if torch.isnan(l_pooled).any():
            raise ValueError("NaN in l_pooled")
                
        if torch.isnan(t5_out).any():
            raise ValueError("NaN in t5_out")

        
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
      
        gc.collect()
        torch.cuda.empty_cache()

        # generate image
        logger.info("Generating image...")
        model = model.to(device)
        print("MODEL DTYPE: ", model.dtype)

        img_ids = img_ids.to(device)
        t5_attn_mask = t5_attn_mask.to(device) if apply_t5_attn_mask else None
        def time_shift(mu: float, sigma: float, t: torch.Tensor):
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


        def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b


        def get_schedule(
            num_steps: int,
            image_seq_len: int,
            base_shift: float = 0.5,
            max_shift: float = 1.15,
            shift: bool = True,
        ) -> list[float]:
            # extra step for zero
            timesteps = torch.linspace(1, 0, num_steps + 1)

            # shifting the schedule to favor high timesteps for higher signal images
            if shift:
                # eastimate mu based on linear estimation between two points
                mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
                timesteps = time_shift(mu, 1.0, timesteps)

            return timesteps.tolist()


        def denoise(
            model,
            img: torch.Tensor,
            img_ids: torch.Tensor,
            txt: torch.Tensor,
            txt_ids: torch.Tensor,
            vec: torch.Tensor,
            timesteps: list[float],
            guidance: float = 4.0,
            t5_attn_mask: Optional[torch.Tensor] = None,
        ):
            # this is ignored for schnell
            guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
            comfy_pbar = comfy.utils.ProgressBar(total=len(timesteps))
            for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
                t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                )
                img = img + (t_prev - t_curr) * pred
                comfy_pbar.update(1)

            return img
        def do_sample(
            accelerator: Optional[accelerate.Accelerator],
            model,
            img: torch.Tensor,
            img_ids: torch.Tensor,
            l_pooled: torch.Tensor,
            t5_out: torch.Tensor,
            txt_ids: torch.Tensor,
            num_steps: int,
            guidance: float,
            t5_attn_mask: Optional[torch.Tensor],
            is_schnell: bool,
            device: torch.device,
            flux_dtype: torch.dtype,
        ):
            timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

            # denoise initial noise
            if accelerator:
                with accelerator.autocast(), torch.no_grad():
                    x = denoise(
                        model, img, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=guidance, t5_attn_mask=t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
                    x = denoise(
                        model, img, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=guidance, t5_attn_mask=t5_attn_mask
                    )

            return x
        
        x = do_sample(accelerator, model, noise, img_ids, l_pooled, t5_out, txt_ids, steps, guidance_scale, t5_attn_mask, False, device, dtype)
        
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # unpack
        x = x.float()
        x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

        # decode
        logger.info("Decoding image...")
        ae = ae.to(device)
        with torch.no_grad():
            if use_fp8:
                with accelerator.autocast():
                    x = ae.decode(x)
            else:
                with torch.autocast(device_type=device.type, dtype=ae_dtype):
                    x = ae.decode(x)

        ae = ae.cpu()

        x = x.clamp(-1, 1)
        x = x.permute(0, 2, 3, 1)

        return ((0.5 * (x + 1.0)).cpu().float(),)   

class UploadToHuggingFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "source_path": ("STRING", {"default": ""}),
                "repo_id": ("STRING",{"default": ""}),
                "revision": ("STRING", {"default": ""}),
                "private": ("BOOLEAN", {"default": True, "tooltip": "If creating a new repo, leave it private"}),
             },
             "optional": {
                "token": ("STRING", {"default": "","tooltip":"DO NOT LEAVE IN THE NODE or it might save in metadata, can also use the hf_token.json"}),
             }
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING",)
    RETURN_NAMES = ("network_trainer","status",)
    FUNCTION = "upload"
    CATEGORY = "FluxTrainer"

    def upload(self, source_path, network_trainer, repo_id, private, revision, token=""):
        with torch.inference_mode(False):
            from huggingface_hub import HfApi
            
            if not token:
                with open(os.path.join(script_directory, "hf_token.json"), "r") as file:
                    token_data = json.load(file)
                token = token_data["hf_token"]
            print(token)

            # Save metadata to a JSON file
            directory_path = os.path.dirname(os.path.dirname(source_path))
            file_name = os.path.basename(source_path)

            metadata = network_trainer["network_trainer"].metadata
            metadata_file_path = os.path.join(directory_path, "metadata.json")
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            repo_type = None
            api = HfApi(token=token)

            try:
                api.repo_info(
                    repo_id=repo_id, 
                    revision=revision if revision != "" else None, 
                    repo_type=repo_type)
                repo_exists = True
                logger.info(f"Repository {repo_id} exists.")
            except Exception as e:  # Catching a more specific exception would be better if you know what to expect
                repo_exists = False
                logger.error(f"Repository {repo_id} does not exist. Exception: {e}")
            
            if not repo_exists:
                try:
                    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
                except Exception as e:  # Checked for RepositoryNotFoundError, but other exceptions could be problematic
                    logger.error("===========================================")
                    logger.error(f"failed to create HuggingFace repo: {e}")
                    logger.error("===========================================")

            is_folder = (type(source_path) == str and os.path.isdir(source_path)) or (isinstance(source_path, Path) and source_path.is_dir())
            print(source_path, is_folder)

            try:
                if is_folder:
                    api.upload_folder(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        folder_path=source_path,
                        path_in_repo=file_name,
                    )
                else:
                    api.upload_file(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        path_or_fileobj=source_path,
                        path_in_repo=file_name,
                    )
                # Upload the metadata file separately if it's not a folder upload
                if not is_folder:
                    api.upload_file(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        path_or_fileobj=str(metadata_file_path),
                        path_in_repo='metadata.json',
                    )
                status = "Uploaded to HuggingFace succesfully"
            except Exception as e:  # RuntimeErrorを確認済みだが他にあると困るので
                logger.error("===========================================")
                logger.error(f"failed to upload to HuggingFace / HuggingFaceへのアップロードに失敗しました : {e}")
                logger.error("===========================================")
                status = f"Failed to upload to HuggingFace {e}"
                
            return (network_trainer, status,)
        
class ExtractFluxLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_model": (folder_paths.get_filename_list("unet"), ),
                "finetuned_model": (folder_paths.get_filename_list("unet"), ),
                "output_path": ("STRING", {"default": f"{str(os.path.join(folder_paths.models_dir, 'loras', 'Flux'))}"}),
                "dim": ("INT", {"default": 4, "min": 2, "max": 1024, "step": 2, "tooltip": "LoRA rank"}),
                "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn"], {"default": "bf16", "tooltip": "the dtype to save the LoRA as"}),
                "load_device": (["cpu", "cuda"], {"default": "cuda", "tooltip": "the device to load the model to"}),
                "store_device": (["cpu", "cuda"], {"default": "cpu", "tooltip": "the device to store the LoRA as"}),
                "clamp_quantile": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "clamp quantile"}),
                "metadata": ("BOOLEAN", {"default": True, "tooltip": "build metadata"}),
                "mem_eff_safe_open": ("BOOLEAN", {"default": False, "tooltip": "memory efficient loading"}),
             },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("output_path",)
    FUNCTION = "extract"
    CATEGORY = "FluxTrainer"

    def extract(self, original_model, finetuned_model, output_path, dim, save_dtype, load_device, store_device, clamp_quantile, metadata, mem_eff_safe_open):
        from .flux_extract_lora import svd
        transformer_path = folder_paths.get_full_path("unet", original_model)
        finetuned_model_path = folder_paths.get_full_path("unet", finetuned_model)
        outpath = svd(
            model_org = transformer_path,
            model_tuned = finetuned_model_path,
            save_to = os.path.join(output_path, f"{finetuned_model.replace('.safetensors', '')}_extracted_lora_rank_{dim}-{save_dtype}.safetensors"),
            dim = dim,
            device = load_device,
            store_device = store_device,
            save_precision = save_dtype,
            clamp_quantile = clamp_quantile,
            no_metadata = not metadata,
            mem_eff_safe_open = mem_eff_safe_open
        )
     
        return (outpath,)

NODE_CLASS_MAPPINGS = {
    "InitFluxLoRATraining": InitFluxLoRATraining,
    "InitFluxTraining": InitFluxTraining,
    "FluxTrainModelSelect": FluxTrainModelSelect,
    "TrainDatasetGeneralConfig": TrainDatasetGeneralConfig,
    "TrainDatasetAdd": TrainDatasetAdd,
    "FluxTrainLoop": FluxTrainLoop,
    "VisualizeLoss": VisualizeLoss,
    "FluxTrainValidate": FluxTrainValidate,
    "FluxTrainValidationSettings": FluxTrainValidationSettings,
    "FluxTrainEnd": FluxTrainEnd,
    "FluxTrainSave": FluxTrainSave,
    "FluxKohyaInferenceSampler": FluxKohyaInferenceSampler,
    "UploadToHuggingFace": UploadToHuggingFace,
    "OptimizerConfig": OptimizerConfig,
    "OptimizerConfigAdafactor": OptimizerConfigAdafactor,
    "FluxTrainSaveModel": FluxTrainSaveModel,
    "ExtractFluxLoRA": ExtractFluxLoRA,
    "OptimizerConfigProdigy": OptimizerConfigProdigy,
    "FluxTrainResume": FluxTrainResume
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InitFluxLoRATraining": "Init Flux LoRA Training",
    "InitFluxTraining": "Init Flux Training",
    "FluxTrainModelSelect": "FluxTrain ModelSelect",
    "TrainDatasetGeneralConfig": "TrainDatasetGeneralConfig",
    "TrainDatasetAdd": "TrainDatasetAdd",
    "FluxTrainLoop": "Flux Train Loop",
    "VisualizeLoss": "Visualize Loss",
    "FluxTrainValidate": "Flux Train Validate",
    "FluxTrainValidationSettings": "Flux Train Validation Settings",
    "FluxTrainEnd": "Flux LoRA Train End",
    "FluxTrainSave": "Flux Train Save LoRA",
    "FluxKohyaInferenceSampler": "Flux Kohya Inference Sampler",
    "UploadToHuggingFace": "Upload To HuggingFace",
    "OptimizerConfig": "Optimizer Config",
    "OptimizerConfigAdafactor": "Optimizer Config Adafactor",
    "FluxTrainSaveModel": "Flux Train Save Model",
    "ExtractFluxLoRA": "Extract Flux LoRA",
    "OptimizerConfigProdigy": "Optimizer Config Prodigy",
    "FluxTrainResume": "Flux Train Resume"
}
