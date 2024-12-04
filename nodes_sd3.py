import os
import torch

import folder_paths
import comfy.model_management as mm
import comfy.utils
import toml
import json
import time
import shutil
import shlex

script_directory = os.path.dirname(os.path.abspath(__file__))

from .sd3_train_network import Sd3NetworkTrainer
from .library import sd3_train_utils as  sd3_train_utils
from .library.device_utils import init_ipex
init_ipex()

from .library import train_util
from .train_network import setup_parser as train_network_setup_parser

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SD3ModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "transformer": (folder_paths.get_filename_list("checkpoints"), ),
                    "clip_l": (folder_paths.get_filename_list("clip"), ),
                    "clip_g": (folder_paths.get_filename_list("clip"), ),
                    "t5": (folder_paths.get_filename_list("clip"), ),
                },
                "optional": {
                    "lora_path": ("STRING",{"multiline": True, "forceInput": True, "default": "", "tooltip": "pre-trained LoRA path to load (network_weights)"}),
                }
        }

    RETURN_TYPES = ("TRAIN_SD3_MODELS",)
    RETURN_NAMES = ("sd3_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer/SD3"

    def loadmodel(self, transformer, clip_l, clip_g, t5, lora_path=""):
        
        transformer_path = folder_paths.get_full_path("checkpoints", transformer)
        clip_l_path = folder_paths.get_full_path("clip", clip_l)
        clip_g_path = folder_paths.get_full_path("clip", clip_g)
        t5_path = folder_paths.get_full_path("clip", t5)

        sd3_models = {
            "transformer": transformer_path,
            "clip_l": clip_l_path,
            "clip_g": clip_g_path,
            "t5": t5_path,
            "lora_path": lora_path
        }
        
        return (sd3_models,)

class InitSD3LoRATraining:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sd3_models": ("TRAIN_SD3_MODELS",),
            "dataset": ("JSON",),
            "optimizer_settings": ("ARGS",),
            "output_name": ("STRING", {"default": "sd35_lora", "multiline": False}),
            "output_dir": ("STRING", {"default": "sd35_trainer_output", "multiline": False, "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "network_dim": ("INT", {"default": 16, "min": 1, "max": 2048, "step": 1, "tooltip": "network dim"}),
            "network_alpha": ("FLOAT", {"default": 16, "min": 0.0, "max": 2048.0, "step": 0.01, "tooltip": "network alpha"}),
            "learning_rate": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "training_shift ": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "shift value for the training distribution of timesteps"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "option for memory use reduction. The maximum number of blocks that can be swapped is 36 for SD3.5L and 22 for SD3.5M"}),
            "fp8_base": ("BOOLEAN", {"default": False, "tooltip": "use fp8 for base model"}),
            "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "fp32", "tooltip": "the actual dtype training uses"}),
            "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16", "tooltip": "the dtype to save checkpoints as"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "sdpa", "tooltip": "memory efficient attention mode"}),
            "train_text_encoder": (['disabled', 'clip_l', 'clip_l_fp8', 'clip_l+T5', 'clip_l+T5_fp8'], {"default": 'disabled', "tooltip": "also train the selected text encoders using specified dtype, T5 can not be trained without clip_l"}),
            "clip_l_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "text encoder learning rate"}),
            "clip_g_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "text encoder learning rate"}),
            "T5_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "text encoder learning rate"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            "gradient_checkpointing": (["enabled", "disabled"], {"default": "enabled", "tooltip": "use gradient checkpointing"}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "additional args to pass to the training command"}),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume args to pass to the training command"}),
                "block_args": ("ARGS", {"default": "", "tooltip": "limit the blocks used in the LoRA"}),
                "loss_args": ("ARGS", {"default": "", "tooltip": "loss args"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS",)
    RETURN_NAMES = ("network_trainer", "epochs_count", "args",)
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/SD3"

    def init_training(self, sd3_models, dataset, optimizer_settings, sample_prompts, output_name, attention_mode, 
                      gradient_dtype, save_dtype, additional_args=None, resume_args=None, train_text_encoder='disabled', 
                      block_args=None, gradient_checkpointing="enabled", prompt=None, extra_pnginfo=None, clip_l_lr=0, clip_g_lr=0, T5_lr=0, loss_args=None, **kwargs):
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
        sd3_train_utils.add_sd3_training_arguments(parser)
        if additional_args is not None:
            print(f"additional_args: {additional_args}")
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
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
            "pretrained_model_name_or_path": sd3_models["transformer"],
            "clip_l": sd3_models["clip_l"],
            "clip_g": sd3_models["clip_g"],
            "t5xxl": sd3_models["t5"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "network_module": ".networks.lora_sd3",
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{save_dtype}",
            "loss_type": "l2",
            "t5xxl_max_token_length": 512,
            "alpha_mask": dataset["alpha_mask"],
            "network_train_unet_only": True if train_text_encoder == 'disabled' else False,
            "fp8_base_unet": True if "fp8" in train_text_encoder else False,
            "disable_mmap_load_safetensors": False,
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

        if train_text_encoder != 'disabled':
            config_dict["text_encoder_lr"] = [clip_l_lr, clip_g_lr, T5_lr]

        #network args
        additional_network_args = []
        
        if "T5" in train_text_encoder:
            additional_network_args.append("train_t5xxl=True")
       
        if block_args:
            additional_network_args.append(block_args["include"])
        
        # Handle network_args in args Namespace
        if hasattr(args, 'network_args') and isinstance(args.network_args, list):
            args.network_args.extend(additional_network_args)
        else:
            setattr(args, 'network_args', additional_network_args)

        if gradient_checkpointing == "disabled":
            config_dict["gradient_checkpointing"] = False
        elif gradient_checkpointing == "enabled_with_cpu_offloading":
            config_dict["gradient_checkpointing"] = True
            config_dict["cpu_offload_checkpointing"] = True
        else:
            config_dict["gradient_checkpointing"] = True

        if sd3_models["lora_path"]:
            config_dict["network_weights"] = sd3_models["lora_path"]

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if loss_args:
            config_dict.update(loss_args)

        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)
        
        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        #workflow saving
        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo["workflow"])
       
        saved_workflow_file_path = os.path.join(output_dir, f"{output_name}_workflow.json")
        with open(saved_workflow_file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        #pass args to kohya and initialize trainer
        with torch.inference_mode(False):
            network_trainer = Sd3NetworkTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)

    
class SD3TrainLoop:
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
            comfy_pbar = comfy.utils.ProgressBar(steps)
            network_trainer.comfy_pbar = comfy_pbar

            network_trainer.optimizer_train_fn()

            while network_trainer.global_step < target_global_step:
                steps_done = training_loop(
                    break_at_steps = target_global_step,
                    epoch = network_trainer.current_epoch.value,
                )
               
                # Also break if the global steps have reached the max train steps
                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break
            
            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, network_trainer.global_step)


class SD3TrainLoRASave:
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


    
class SD3TrainEnd:
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
    OUTPUT_NODE = True

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]
            
            network_trainer.metadata["ss_epoch"] = str(network_trainer.num_train_epochs)
            network_trainer.metadata["ss_training_finished_at"] = str(time.time())

            network = network_trainer.accelerator.unwrap_model(network_trainer.network)

            network_trainer.accelerator.end_training()
            network_trainer.optimizer_eval_fn()

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
    
class SD3TrainValidationSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 20, "min": 1, "max": 256, "step": 1, "tooltip": "sampling steps"}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "image width"}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "image height"}),
            "guidance_scale": ("FLOAT", {"default": 4, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "guidance scale"}),
            "seed": ("INT", {"default": 42,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
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
        
class SD3TrainValidate:
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
            network_trainer.current_epoch.value, 
            network_trainer.global_step,
            validation_settings
        )
        network_trainer.optimizer_eval_fn()
        with torch.inference_mode(False):
            image_tensors = network_trainer.sample_images(*params)

        
        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, (0.5 * (image_tensors + 1.0)).cpu().float(),)
    
NODE_CLASS_MAPPINGS = {
    "SD3ModelSelect": SD3ModelSelect,
    "InitSD3LoRATraining": InitSD3LoRATraining,
    "SD3TrainValidationSettings": SD3TrainValidationSettings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3ModelSelect": "SD3 Model Select",
    "InitSD3LoRATraining": "Init SD3 LoRA Training",
    "SD3TrainValidationSettings": "SD3 Train Validation Settings",
}
