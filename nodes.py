import os
import torch
from torchvision import transforms
import math
import copy
import folder_paths
import comfy.model_management as mm
import comfy.utils
import argparse
from typing import Any, List
import time

script_directory = os.path.dirname(os.path.abspath(__file__))

import torch
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='bf16', cpu=False)
from .library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from .library import flux_models, flux_train_utils, flux_utils, sd3_train_utils, strategy_base, strategy_flux, train_util
from .train_network import NetworkTrainer, setup_parser

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs"

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def load_target_model(self, args, weight_dtype, accelerator):
        # currently offload to cpu for some models
        name = "schnell" if "schnell" in args.pretrained_model_name_or_path else "dev"  # TODO change this to a more robust way
        # if we load to cpu, flux.to(fp8) takes a long time
        model = flux_utils.load_flow_model(name, args.pretrained_model_name_or_path, weight_dtype, "cpu")

        if args.split_mode:
            model = self.prepare_split_model(model, weight_dtype, accelerator, args)

        clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu")
        clip_l.eval()

        # loading t5xxl to cpu takes a long time, so we should load to gpu in future
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, weight_dtype, "cpu")
        t5xxl.eval()

        ae = flux_utils.load_ae(name, args.ae, weight_dtype, "cpu")

        return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model

    def prepare_split_model(self, model, weight_dtype, accelerator, args):
        from accelerate import init_empty_weights

        logger.info("prepare split model")
        with init_empty_weights():
            flux_upper = flux_models.FluxUpper(model.params)
            flux_lower = flux_models.FluxLower(model.params)
        sd = model.state_dict()

        # lower (trainable)
        logger.info("load state dict for lower")
        flux_lower.load_state_dict(sd, strict=False, assign=True)
        flux_lower.to(dtype=weight_dtype)

        # upper (frozen)
        logger.info("load state dict for upper")
        flux_upper.load_state_dict(sd, strict=False, assign=True)

        logger.info("prepare upper model")
        target_dtype = torch.float8_e4m3fn if args.fp8_base else weight_dtype
        flux_upper.to(accelerator.device, dtype=target_dtype)
        flux_upper.eval()

        if args.fp8_base:
            # this is required to run on fp8
            flux_upper = accelerator.prepare(flux_upper)

        flux_upper.to("cpu")

        self.flux_upper = flux_upper
        del model  # we don't need model anymore
        clean_memory_on_device(accelerator.device)

        logger.info("split model prepared")

        return flux_lower

    def get_tokenize_strategy(self, args):
        return strategy_flux.FluxTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        return [tokenize_strategy.clip_l, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders  # + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, False, apply_t5_attn_mask=args.apply_t5_attn_mask
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # reduce memory consumption
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator.is_main_process)

            # cache sample prompts
            self.sample_prompts_te_outputs = None
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = []
                for line in args.sample_prompts:
                    line = line.strip()
                    if len(line) > 0 and line[0] != "#":
                        prompts.append(line)
                
                # preprocess prompts
                for i in range(len(prompts)):
                    prompt_dict = prompts[i]
                    if isinstance(prompt_dict, str):
                        from library.train_util import line_to_prompt_dict

                        prompt_dict = line_to_prompt_dict(prompt_dict)
                        prompts[i] = prompt_dict
                    assert isinstance(prompt_dict, dict)

                    # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
                    prompt_dict["enum"] = i
                    prompt_dict.pop("subset", None)

                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks, args.apply_t5_attn_mask
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs
            accelerator.wait_for_everyone()

            logger.info("move text encoders back to cpu")
            text_encoders[0].to("cpu")  # , dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu")  # , dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoder
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def sample_images(self, accelerator, args, epoch, global_step, ae, text_encoder, flux, validation_settings):
        if not args.split_mode:
            image_tensors = flux_train_utils.sample_images(
                accelerator, args, epoch, global_step, flux, ae, text_encoder, self.sample_prompts_te_outputs, validation_settings
            )
            return image_tensors

        class FluxUpperLowerWrapper(torch.nn.Module):
            def __init__(self, flux_upper: flux_models.FluxUpper, flux_lower: flux_models.FluxLower, device: torch.device):
                super().__init__()
                self.flux_upper = flux_upper
                self.flux_lower = flux_lower
                self.target_device = device

            def forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
                self.flux_lower.to("cpu")
                clean_memory_on_device(self.target_device)
                self.flux_upper.to(self.target_device)
                img, txt, vec, pe = self.flux_upper(img, img_ids, txt, txt_ids, timesteps, y, guidance)
                self.flux_upper.to("cpu")
                clean_memory_on_device(self.target_device)
                self.flux_lower.to(self.target_device)
                return self.flux_lower(img, txt, vec, pe)

        wrapper = FluxUpperLowerWrapper(self.flux_upper, flux, accelerator.device)
        clean_memory_on_device(accelerator.device)
        flux_train_utils.sample_images(
            accelerator, args, epoch, global_step, flux, ae, text_encoder, self.sample_prompts_te_outputs, validation_settings
        )
        clean_memory_on_device(accelerator.device)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler
    
    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs

    def encode_images_to_latents(self, args, accelerator, vae, images):
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
    ):
        # copy from sd3_train.py and modified

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = self.noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        def compute_density_for_timestep_sampling(
            weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
        ):
            """Compute the density for sampling the timesteps when doing SD3 training.

            Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

            SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
            """
            if weighting_scheme == "logit_normal":
                # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
                u = torch.nn.functional.sigmoid(u)
            elif weighting_scheme == "mode":
                u = torch.rand(size=(batch_size,), device="cpu")
                u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
            else:
                u = torch.rand(size=(batch_size,), device="cpu")
            return u

        def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
            """Computes loss weighting scheme for SD3 training.

            Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

            SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
            """
            if weighting_scheme == "sigma_sqrt":
                weighting = (sigmas**-2.0).float()
            elif weighting_scheme == "cosmap":
                bot = 1 - 2 * sigmas + 2 * sigmas**2
                weighting = 2 / (math.pi * bot)
            else:
                weighting = torch.ones_like(sigmas)
            return weighting

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
            # Simple random t-based noise sampling
            if args.timestep_sampling == "sigmoid":
                # https://github.com/XLabs-AI/x-flux/tree/main
                t = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=accelerator.device))
            else:
                t = torch.rand((bsz,), device=accelerator.device)
            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

            # Add noise according to flow matching.
            sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=weight_dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # get guidance
        guidance_vec = torch.full((bsz,), args.guidance_scale, device=accelerator.device)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)
            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        # Predict the noise residual
        l_pooled, t5_out, txt_ids = text_encoder_conds
        # print(
        #     f"model_input: {noisy_model_input.shape}, img_ids: {img_ids.shape}, t5_out: {t5_out.shape}, txt_ids: {txt_ids.shape}, l_pooled: {l_pooled.shape}, timesteps: {timesteps.shape}, guidance_vec: {guidance_vec.shape}"
        # )

        if not args.split_mode:
            # normal forward
            with accelerator.autocast():
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                model_pred = unet(
                    img=packed_noisy_model_input,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                )
        else:
            # split forward to reduce memory usage
            assert network.train_blocks == "single", "train_blocks must be single for split mode"
            with accelerator.autocast():
                # move flux lower to cpu, and then move flux upper to gpu
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)
                self.flux_upper.to(accelerator.device)

                # upper model does not require grad
                with torch.no_grad():
                    intermediate_img, intermediate_txt, vec, pe = self.flux_upper(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000,
                        guidance=guidance_vec,
                    )

                # move flux upper back to cpu, and then move flux lower to gpu
                self.flux_upper.to("cpu")
                clean_memory_on_device(accelerator.device)
                unet.to(accelerator.device)

                # lower model requires grad
                intermediate_img.requires_grad_(True)
                intermediate_txt.requires_grad_(True)
                vec.requires_grad_(True)
                pe.requires_grad_(True)
                model_pred = unet(img=intermediate_img, txt=intermediate_txt, vec=vec, pe=pe)

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        if args.model_prediction_type == "raw":
            # use model_pred as is
            weighting = None
        elif args.model_prediction_type == "additive":
            # add the model_pred to the noisy_model_input
            model_pred = model_pred + noisy_model_input
            weighting = None
        elif args.model_prediction_type == "sigma_scaled":
            # apply sigma scaling
            model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        return model_pred, target, timesteps, None, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, flux="dev")

    def update_metadata(self, metadata, args):
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

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
            "flip_aug": ("BOOLEAN",{"default": False},{"tooltip": "enable horizontal flip augmentation"}),
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
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "split_mode": ("BOOLEAN", {"default": False, "tooltip": "[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"}),
            "weighting_scheme": (["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],),
            "logit_mean": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean to use when using the logit_normal weighting scheme"}),
            "logit_std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "std to use when using the logit_normal weighting scheme"}),
            "mode_scale": ("FLOAT", {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scale of mode weighting scheme. Only effective when using the mode as the weighting_scheme"}),
            "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.01, "tooltip": "the FLUX.1 dev variant is a guidance distilled model"}),
            "timestep_sampling": (["sigmoid", "uniform", "sigma"], {"tooltip": "method to sample timestep"}),
            "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for sigmoid timestep sampling (only used when timestep-sampling is sigmoid"}),
            "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"tooltip": "How to interpret and process the model prediction: raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "for the Euler Discrete Scheduler, default is 3.0"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "default", "tooltip": "memory efficient attention mode"}),
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
            "fp8_base": True,
            "dataset_config": dataset,
            "output_dir": output_dir,
            "output_name": output_name,
            "loss_type": "l2",
            "optimizer_type": optimizer_type,
        }
        attention_settings = {
            "sdpa": {"mem_eff_attn": True, "xformers": False, "spda": True},
            "xformers": {"mem_eff_attn": True, "xformers": True, "spda": False}
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
            "end": ("BOOLEAN", {"default": False, "tooltip": "whether to end training"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER",)
    RETURN_NAMES = ("network_trainer",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer"

    def train(self, network_trainer, steps, end):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]
            initial_global_step = network_trainer.global_step

            target_global_step = network_trainer.global_step + steps
            pbar = comfy.utils.ProgressBar(steps)
            while network_trainer.global_step < target_global_step:
                epoch = training_loop(
                    break_at_steps=target_global_step,
                    epoch=network_trainer.current_epoch.value,
                    num_train_epochs=network_trainer.num_train_epochs,
                    accelerator=network_trainer.accelerator,
                    network=network_trainer.network,
                    text_encoder=network_trainer.text_encoder,
                    unet=network_trainer.unet,
                    vae=network_trainer.vae,
                    tokenizers=network_trainer.tokenizers,
                    args=network_trainer.args,
                    train_dataloader=network_trainer.train_dataloader,
                    initial_step=network_trainer.initial_step,
                    global_step=network_trainer.global_step,
                    current_epoch=network_trainer.current_epoch,
                    metadata=network_trainer.metadata,
                    optimizer=network_trainer.optimizer,
                    lr_scheduler=network_trainer.lr_scheduler,
                    loss_recorder=network_trainer.loss_recorder
                )
                pbar.update(network_trainer.global_step - initial_global_step)
               
                # Also break if the global steps have reached the max train steps
                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break
            
            # with torch.inference_mode(True):
            #     image_tensors = network_trainer.sample_images(
            #         network_trainer.accelerator, 
            #         network_trainer.args, 
            #         epoch, 
            #         network_trainer.global_step,
            #         network_trainer.vae,
            #         network_trainer.text_encoder,
            #         network_trainer.unet
            #         )
            #     print(image_tensors.min(), image_tensors.max(), image_tensors.shape)

            if end:
                network_trainer.metadata["ss_epoch"] = str(network_trainer.num_train_epochs)
                network_trainer.metadata["ss_training_finished_at"] = str(time.time())

                network = network_trainer.accelerator.unwrap_model(network_trainer.network)

                network_trainer.accelerator.end_training()

                train_util.save_state_on_train_end(network_trainer.args, network_trainer.accelerator)
                ckpt_name = train_util.get_last_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as)
                network_trainer.save_model(ckpt_name, network, network_trainer.global_step, network_trainer.num_train_epochs, force_sync_upload=True)
                logger.info("model saved.")
            else:
                ckpt_name = train_util.get_epoch_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as, epoch + 1)
                network_trainer.save_model(ckpt_name, accelerator.unwrap_model(network_trainer.network), network_trainer.global_step, epoch + 1)

                remove_epoch_no = train_util.get_remove_epoch_no(network_trainer.args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_util.get_epoch_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as, remove_epoch_no)
                    network_trainer.remove_model(remove_ckpt_name)

                if network_trainer.args.save_state:
                    train_util.save_and_remove_state_on_epoch_end(network_trainer.args, accelerator, epoch + 1)

            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, )

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
    "FluxTrainValidationSettings": FluxTrainValidationSettings
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InitFluxTraining": "Init Flux Training",
    "FluxTrainModelSelect": "FluxTrain ModelSelect",
    "TrainDatasetConfig": "Train Dataset Config",
    "FluxTrainLoop": "Flux Train Loop",
    "VisualizeLoss": "Visualize Loss",
    "FluxTrainValidate": "Flux Train Validate",
    "FluxTrainValidationSettings": "Flux Train Validation Settings"
}
