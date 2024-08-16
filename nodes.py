import os
import torch
from torchvision import transforms

import folder_paths
import comfy.model_management as mm
import comfy.utils

import time

script_directory = os.path.dirname(os.path.abspath(__file__))

from .flux_train_network_comfy import FluxNetworkTrainer

from .library.device_utils import init_ipex
init_ipex()

from .library import train_util
from .train_network import setup_parser

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

class TrainDatasetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT",{"min": 64, "default": 512}),
            "height": ("INT",{"min": 64, "default": 512}),
            "batch_size": ("INT",{"min": 1, "default": 2}),
            "dataset_path": ("STRING",{"multiline": True, "default": ""}),
            "class_tokens": ("STRING",{"multiline": True, "default": ""}),
            "enable_bucket": ("BOOLEAN",{"default": True, "tooltip": "enable buckets for multi aspect ratio training"}),
            "bucket_no_upscale": ("BOOLEAN",{"default": False, "tooltip": "bucket reso is defined by image size automatically"}),
            "min_bucket_reso": ("INT",{"min": 64, "default": 256}),
            "max_bucket_reso": ("INT",{"min": 64, "default": 1024}),
            "color_aug": ("BOOLEAN",{"default": False, "tooltip": "enable weak color augmentation"}),
            "flip_aug": ("BOOLEAN",{"default": False, "tooltip": "enable horizontal flip augmentation"}),
            },
        }

    RETURN_TYPES = ("TOML_DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, dataset_path, class_tokens, width, height, batch_size, enable_bucket, color_aug, flip_aug, 
                  bucket_no_upscale, min_bucket_reso, max_bucket_reso):
        import toml

        dataset = {
           "general": {
               "shuffle_caption": False,
               "caption_extension": ".txt",
           },
           "datasets": [
               {
                   "resolution": (width, height),
                   "batch_size": batch_size,  
                   "keep_tokens": 2,
                   "enable_bucket": enable_bucket,
                   "bucket_no_upscale": bucket_no_upscale,
                   "min_bucket_reso": min_bucket_reso,
                   "max_bucket_reso": max_bucket_reso,
                   "color_aug": color_aug,
                   "flip_aug": flip_aug,
                   "subsets": [
                       {
                           "image_dir": dataset_path,
                           "class_tokens": class_tokens
                       }
                   ]
               }
           ]
        }
        
        return (toml.dumps(dataset),)
    
class InitFluxTraining:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset": ("TOML_DATASET",),
            "output_name": ("STRING", {"default": "train_flux", "multiline": False}),
            "network_dim": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1, "tooltip": "network dim"}),
            "learning_rate": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.00001, "tooltip": "learning rate"}),
            "unet_lr": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.00001, "tooltip": "unet learning rate"}),
            #"max_train_epochs": ("INT", {"default": 4, "min": 1, "max": 1000, "step": 1, "tooltip": "max number of training epochs"}),
            "optimizer_type": (["adamw8bit", "adafactor", "prodigy"], {"default": "adamw8bit", "tooltip": "optimizer type"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 10000, "step": 1, "tooltip": "max number of training steps"}),
            "network_train_unet_only": ("BOOLEAN", {"default": True, "tooltip": "wheter to train the text encoder"}),
            "text_encoder_lr": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.00001, "tooltip": "text encoder learning rate"}),
            "apply_t5_attn_mask": ("BOOLEAN", {"default": True, "tooltip": "apply t5 attention mask"}),
            "t5xxl_max_token_length": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "dev uses 512, schnell 256"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "split_mode": ("BOOLEAN", {"default": False, "tooltip": "[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"}),
            "weighting_scheme": (["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],),
            "logit_mean": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean to use when using the logit_normal weighting scheme"}),
            "logit_std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "std to use when using the logit_normal weighting scheme"}),
            "mode_scale": ("FLOAT", {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scale of mode weighting scheme. Only effective when using the mode as the weighting_scheme"}),
            "timestep_sampling": (["sigmoid", "uniform", "sigma"], {"tooltip": "method to sample timestep"}),
            "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for sigmoid timestep sampling (only used when timestep-sampling is sigmoid"}),
            "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"tooltip": "How to interpret and process the model prediction: raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "for the Euler Discrete Scheduler, default is 3.0"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "fp8_base": ("BOOLEAN", {"default": True, "tooltip": "use fp8 for base model"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "sdpa", "tooltip": "memory efficient attention mode"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "STRING", )
    RETURN_NAMES = ("network_trainer", "epochs_count", "output_path",)
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, dataset, sample_prompts, output_name, optimizer_type, attention_mode, **kwargs,):
        mm.soft_empty_cache()

        parser = setup_parser()
        args = parser.parse_args()

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

        #dataset_config = os.path.join(script_directory, "dataset_flux.toml")
        output_dir = os.path.join(script_directory, "output")
        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        config_dict = {
            "sample_prompts": prompts,
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
            "save_precision": "bf16",
            "network_module": "networks.lora_flux",
            "dataset_config": dataset,
            "output_dir": output_dir,
            "output_name": output_name,
            "loss_type": "l2",
            "optimizer_type": optimizer_type,
            "guidance_scale": 3.5,
        }
        attention_settings = {
            "sdpa": {"mem_eff_attn": True, "xformers": False, "sdpa": True},
            "xformers": {"mem_eff_attn": True, "xformers": True, "sdpa": False}
        }
        config_dict.update(attention_settings.get(attention_mode, {}))

        if optimizer_type == "adafactor":
            config_dict["optimizer_args"] = [
                "relative_step=False",
                "scale_parameter=False",
                "warmup_init=False"
            ]
        config_dict.update(kwargs)

        for key, value in config_dict.items():
            setattr(args, key, value)

        with torch.inference_mode(False):
            network_trainer = FluxNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        final_output_lora_path = os.path.join(output_dir, "output", output_name)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, final_output_lora_path)
    
class FluxTrainLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER",)
    RETURN_NAMES = ("network_trainer",)
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
        return (trainer, )

class FluxTrainSave:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": False}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING",)
    RETURN_NAMES = ("network_trainer","lora_path",)
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer"

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            
            ckpt_name = train_util.get_epoch_ckpt_name(trainer.args, "." + trainer.args.save_model_as, trainer.current_epoch.value + 1)
            trainer.save_model(ckpt_name, trainer.accelerator.unwrap_model(trainer.network), trainer.global_step, trainer.current_epoch.value + 1)

            remove_epoch_no = train_util.get_remove_epoch_no(trainer.args, trainer.current_epoch.value + 1)
            if remove_epoch_no is not None:
                remove_ckpt_name = train_util.get_epoch_ckpt_name(trainer.args, "." + trainer.args.save_model_as, remove_epoch_no)
                trainer.remove_model(remove_ckpt_name)

            if save_state:
                train_util.save_and_remove_state_on_epoch_end(trainer.args, trainer.accelerator, trainer.current_epoch.value + 1)

            lora_path = os.path.join(trainer.args.output_dir, "output", ckpt_name)
            
        return (network_trainer, lora_path)
    
class FluxTrainEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": True}),
             },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
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

            final_output_lora_path = os.path.join(network_trainer.args.output_dir, "output", network_trainer.args.output_name)

            training_loop = None
            network_trainer = None
            mm.soft_empty_cache()
            
        return (final_output_lora_path,)
    
class FluxTrainValidationSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 20, "min": 1, "max": 256, "step": 1, "tooltip": "sampling steps"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image width"}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image height"}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "guidance scale"}),
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

        with torch.inference_mode(True):
            image_tensors = network_trainer.sample_images(
                network_trainer.accelerator, 
                network_trainer.args, 
                network_trainer.current_epoch.value, 
                network_trainer.global_step,
                network_trainer.vae,
                network_trainer.text_encoder,
                network_trainer.unet,
                validation_settings
                )

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
             },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot",)
    FUNCTION = "draw"
    CATEGORY = "FluxTrainer"

    def draw(self, network_trainer):
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        # Example list of loss values
        loss_values = network_trainer["network_trainer"].loss_recorder.loss_list

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot(loss_values, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True)

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Convert the BytesIO object to a PIL Image
        image = Image.open(buf).convert('RGB')

        # Convert the PIL Image to a torch tensor
        image_tensor = transforms.ToTensor()(image)
        print(image_tensor.shape)
        image_tensor = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        print(image_tensor.shape)

        return image_tensor,

    

NODE_CLASS_MAPPINGS = {
    "InitFluxTraining": InitFluxTraining,
    "FluxTrainModelSelect": FluxTrainModelSelect,
    "TrainDatasetConfig": TrainDatasetConfig,
    "FluxTrainLoop": FluxTrainLoop,
    "VisualizeLoss": VisualizeLoss,
    "FluxTrainValidate": FluxTrainValidate,
    "FluxTrainValidationSettings": FluxTrainValidationSettings,
    "FluxTrainEnd": FluxTrainEnd,
    "FluxTrainSave": FluxTrainSave
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InitFluxTraining": "Init Flux Training",
    "FluxTrainModelSelect": "FluxTrain ModelSelect",
    "TrainDatasetConfig": "Train Dataset Config",
    "FluxTrainLoop": "Flux Train Loop",
    "VisualizeLoss": "Visualize Loss",
    "FluxTrainValidate": "Flux Train Validate",
    "FluxTrainValidationSettings": "Flux Train Validation Settings",
    "FluxTrainEnd": "Flux Train End",
    "FluxTrainSave": "Flux Train Save"
}
