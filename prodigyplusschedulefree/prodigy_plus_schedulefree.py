# source: @LoganBooker - https://github.com/LoganBooker/prodigy-plus-schedule-free
import torch
from .core_optimiser import CoreOptimiser

class ProdigyPlusScheduleFree(CoreOptimiser):
    r"""
    An optimiser based on Prodigy that includes schedule-free logic. Has additional improvements in the form of optional StableAdamW 
    gradient scaling and Adam-atan2 updates, per parameter group adaptation, lower memory utilisation, fused back pass support and 
    tweaks to mitigate uncontrolled LR growth.

    Based on code from:
    https://github.com/facebookresearch/schedule_free
    https://github.com/konstmish/prodigy

    Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
    https://github.com/konstmish/prodigy/pull/23
    https://github.com/konstmish/prodigy/pull/22
    https://github.com/konstmish/prodigy/pull/20

    As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
    calls to `train()` and `eval()`. See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free
    
    If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

    Leave `lr` set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
    ability for the optimiser to predict stepsizes. Gradient clipping/normalisation is already handled in the following configurations:
    
    1) `use_stableadamw=True,eps=1e8` (or any reasonable positive epsilon)
    2) `eps=None` (Adam-atan2, scale invariant and can mess with Prodigy's stepsize calculations in some scenarios)

    A new parameter, `beta4`, allows `d` to be updated via a moving average, rather than being immediately updated. This can help
    smooth out learning rate adjustments. Values of 0.9-0.99 are recommended if trying out the feature. If set to None, the 
    square root of `beta1` is used, while a setting of 0 (the default) disables the feature.

    By default, `split_groups` is set to `True`, so each parameter group will have its own adaptation values. So if you're training
    different networks together, they won't contaminate each other's learning rates. The disadvantage of this approach is that some 
    networks can take a long time to reach a good learning rate when trained alongside others (for example, SDXL's Unet). 
    It's recommended to use a higher `d0` (1e-5, 5e-5, 1e-4) so these networks don't get stuck at a low learning rate.
    
    For Prodigy's reference behaviour, which lumps all parameter groups together, set `split_groups` to `False`.

    In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
    can be controlled via the `prodigy_steps` settings.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): 
            Coefficients used for computing running averages of gradient and its square 
            (default: (0.9, 0.99))
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. If set to None,
            Adam-atan2 is used instead. This removes the need for epsilon tuning, but may not work well in all situations.
            (default: 1e-8).
        beta3 (float):
            Coefficient for computing the Prodigy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        beta4 (float):
            Coefficient for updating the learning rate from Prodigy's adaptive stepsize. Smooths out spikes in learning rate adjustments. 
            If set to None, beta1 is used instead. (default 0, which disables smoothing and uses original Prodigy behaviour).
        weight_decay (float):
            Decoupled weight decay. Value is multiplied by the adaptive learning rate.
            (default: 0).
        use_bias_correction (boolean):
            Turn on Adafactor-style bias correction, which scales beta2 directly. (default False).
        d0 (float):
            Initial estimate for Prodigy (default 1e-6).
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0). Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        prodigy_steps (int):
            Freeze Prodigy stepsize adjustments after a certain optimiser step.
            (default 0)
        warmup_steps (int):
            Enables a linear learning rate warmup (default 0). Use this over the warmup settings of your LR scheduler.
        split_groups (boolean):
            Track individual adaptation values for each parameter group. For example, if training
            a text encoder beside a Unet. Note this can have a significant impact on training dynamics.
            Set to False for original Prodigy behaviour, where all groups share the same values.
            (default True)
        split_groups_mean (str: None, "mean", "harmonic_mean", "geometric_mean"):
            When split_groups is True, use specified mean of learning rates for all groups. This favours
            a more conservative LR. Calculation remains per-group. If split_groups is False, this value has no effect.
            Set to None to have each group use its own learning rate calculation. 
            (default "harmonic_mean")
        factored (boolean):
            Use factored approximation of the second moment, similar to Adafactor. Reduces memory usage. Disable
            if training results in NaNs or the learning rate fails to grow.
            (default True)
        fused_back_pass (boolean):
            Stops the optimiser from running the normal step method. Set to True if using fused backward pass.
            (default False)
        use_stableadamw (boolean):
            Scales parameter updates by the root-mean-square of the normalised gradient, in essence identical to 
            Adafactor's gradient scaling. Set to False if the adaptive learning rate never improves.
            (default True)
        use_muon_pp (boolean):
            Experimental. Perform orthogonalisation post-processing on 2D+ parameter updates ala Shampoo/SOAP/Muon.
            (https://github.com/KellerJordan/Muon/blob/master/muon.py). Not suitable for all training scenarios.
            May not work well with small batch sizes or finetuning. (default False)
        use_cautious (boolean):
            Experimental. Perform "cautious" updates, as proposed in https://arxiv.org/pdf/2411.16085. Modifies
            the update to isolate and boost values that align with the current gradient.
            (default False)
        use_adopt (boolean):
            Experimental. Performs a modified step where the second moment is updated after the parameter update,
            so as not to include the current gradient in the denominator. This is a partial implementation of ADOPT 
            (https://arxiv.org/abs/2411.02853), as we don't have a first moment to use for the update.
            (default False)
        stochastic_rounding (boolean):
            Use stochastic rounding for bfloat16 weights (https://github.com/pytorch/pytorch/issues/120376). Brings
            bfloat16 training performance close to that of float32.
            (default True)
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.99), beta3=None, beta4=0,
                 weight_decay=0.0,
                 use_bias_correction=False,
                 d0=1e-6, d_coef=1.0,
                 prodigy_steps=0,
                 warmup_steps=0,
                 eps=1e-8,
                 split_groups=True,
                 split_groups_mean="harmonic_mean",
                 factored=True,
                 fused_back_pass=False,
                 use_stableadamw=True,
                 use_muon_pp=False,
                 use_cautious=False,
                 use_adopt=False,
                 stochastic_rounding=True):

        super().__init__(params=params, lr=lr, betas=betas, beta3=beta3, beta4=beta4,
                        weight_decay=weight_decay, use_bias_correction=use_bias_correction,
                        d0=d0, d_coef=d_coef, prodigy_steps=prodigy_steps,
                        warmup_steps=warmup_steps, eps=eps, split_groups=split_groups,
                        split_groups_mean=split_groups_mean, factored=factored,
                        fused_back_pass=fused_back_pass, use_stableadamw=use_stableadamw,
                        use_muon_pp=use_muon_pp, use_cautious=use_cautious, use_adopt=use_adopt,
                        stochastic_rounding=stochastic_rounding)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group['train_mode']:
                continue
            beta1, _ = group['betas']
            for p in group['params']:
                z = self.state[p].get('z')
                if z is not None:
                    # Set p to x
                    p.lerp_(end=z.to(device=p.device), weight=1 - 1 / beta1)
            group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group['train_mode']:
                continue
            beta1, _ = group['betas']
            for p in group['params']:
                z = self.state[p].get('z')
                if z is not None:
                    # Set p to y
                    p.lerp_(end=z.to(device=p.device), weight=1 - beta1)
            group['train_mode'] = True

    @torch.no_grad()
    def initialise_state(self, p, factored, use_muon_pp):
        state, needs_init = self.initialise_state_internal(p, factored, use_muon_pp)

        if needs_init:
            state['z'] = p.detach().clone(memory_format=torch.preserve_format)        
        
        return state
    
    @torch.no_grad()
    def update_params(self, y, z, update, dlr, group):
        # Weight decay.
        weight_decay = group['weight_decay']

        if weight_decay != 0:
            update.add_(y, alpha=weight_decay)

        weight = dlr ** 2
        weight_sum = group['weight_sum'] + weight
        ckp1 = weight / weight_sum if weight_sum else 0

        y.lerp_(end=z, weight=ckp1)
        y.add_(update, alpha=dlr * (group['betas'][0] * (1 - ckp1) - 1))
        z.sub_(update, alpha=dlr)

        return weight_sum

    @torch.no_grad()
    def step_param(self, p, group):
        if not group['train_mode']:
            raise Exception("Not in train mode!")

        self.on_start_step(group)

        weight_sum = group['weight_sum']
        
        if p.grad is not None:
            grad = p.grad

            state = self.initialise_state(p, group['factored'], group['use_muon_pp'])
            use_adopt = group['use_adopt']

            if use_adopt and group['k'] == 1:
                self.update_second_moment(state, group, grad.float(), 0, return_denom=False)
            else:
                dlr = self.get_dlr(group)
                rms_min = 1.0 if group['use_stableadamw'] else None
                y, z = p, state['z']

                self.update_prodigy(state, group, grad, z, dlr)

                grad_mask = grad.clone() if group['use_cautious'] else None

                if state['muon']:
                    # newton_schulz_ casts to bf16 internally, so do float cast afterwards.
                    update = self.newton_schulz_(grad).float()
                    rms_min = 1e-30
                else:
                    grad = grad.float()
                    _, beta2 = group['betas']

                    if use_adopt:
                        denom = self.get_denom(state, group)
                        self.update_second_moment(state, group, grad, beta2, return_denom=False)
                    else:
                        denom = self.update_second_moment(state, group, grad, beta2)

                    update = self.get_update(grad, denom, group)
                    del denom

                    if group['eps'] is None:
                        rms_min = None

                self.rms_(update, rms_min)

                if grad_mask is not None:
                    self.cautious_(update, grad_mask, reuse_grad=True)

                if group['stochastic_rounding'] and y.dtype == z.dtype == torch.bfloat16:
                    y_fp32, z_fp32 = y.float(), z.float()

                    weight_sum = self.update_params(y_fp32, z_fp32, update, dlr, group)

                    self.copy_stochastic_(y, y_fp32)
                    self.copy_stochastic_(z, z_fp32)

                    del y_fp32, z_fp32
                else:
                    weight_sum = self.update_params(y, z, update, dlr, group)

                del update

        if self.on_end_step(group):
            group['weight_sum'] = weight_sum