# source: @LoganBooker - https://github.com/LoganBooker/prodigy-plus-schedule-free
import torch
from .core_optimiser import CoreOptimiser

class ProdigyPlusScheduleFree(CoreOptimiser):
    r"""
    An optimiser based on Prodigy that includes schedule-free logic. Has additional improvements in the form of optional StableAdamW 
    gradient scaling and Adam-atan2 updates, per parameter group adaptation, lower memory utilisation and fused back pass support.

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
    2) `eps=None` (Adam-atan2, scale invariant. Will disable StableAdamW if enabled.)

    By default, `split_groups` and `split_groups_mean` are set to `True`, so each parameter group will have its own `d` values, however,
    they will all use the harmonic mean for the dynamic learning rate. To make each group use its own dynamic LR, set `split_groups_mean` to False.
    To use the reference Prodigy behaviour where all groups are combined, set `split_groups` to False. 
    
    In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
    can be controlled via the `prodigy_steps` settings. This will also free any Prodigy-specific memory used by the
    optimiser (though with all the memory-related improvements, this should not be significant unless you're training
    very large models).

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
            (default: 1.0)
        betas (Tuple[float, float], optional): 
            Coefficients used for computing running averages of gradient and its square.
            (default: (0.9, 0.99))
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. If set to None,
            Adam-atan2 is used instead. This removes the need for epsilon tuning, but may not work well in all situations.
            (default: 1e-8).
        beta3 (float):
            Coefficient for computing the Prodigy stepsize using running averages. If set to None, uses the value of 
            square root of beta2 
            (default: None).
        weight_decay (float):
            Decoupled weight decay. Use the weight_decay_by_lr setting to determine if decay should be multiplied by the
            adaptive learning rate.
            (default: 0).
        weight_decay_by_lr (boolean):
            If True, weight_decay is multiplied by the adaptive learning rate (as per the PyTorch implementation of AdamW).
            If False, weight_decay will have a much stronger effect.
            (default: True).
        use_bias_correction (boolean):
            Use the RAdam variant of schedule-free (https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/radam_schedulefree.py).
            This combines bias correction with automatic warmup. Please note this will significantly dampen Prodigy's adaptive stepsize
            calculations -- it can take up to 10 times longer to start adjusting the learning rate. This can be mitigated somewhat by enabling
            SPEED (use_speed=True).
            (default: False).
        d0 (float):
            Initial estimate for Prodigy. Also serves as the minimum learning rate.
            (default: 1e-6).
        d_coef (float):
            Coefficient in the expression for the estimate of d. Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
            (default: 1.0)
        prodigy_steps (int):
            Freeze Prodigy stepsize adjustments after a certain optimiser step and releases all state memory required
            by Prodigy.
            (default: 0)
        use_speed (boolean):
            Highly experimental. Signed Prodigy with ExponEntial D. This decouples the adaptive stepsize calculations from 
            the magnitude of the weights and gradient. This can provide faster, more accurate LRs in some scenarios, 
            but may fail in situations where the optimal LR is very close to (or less than) d0.
            (default: False):
        split_groups (boolean):
            Track individual adaptation values for each parameter group. For example, if training
            a text encoder beside a Unet. Note this can have a significant impact on training dynamics.
            Set to False for original Prodigy behaviour, where all groups share the same values.
            (default: True)
        split_groups_mean (boolean):
            When split_groups is True, use the harmonic mean of learning rates for all groups. This favours
            a more conservative LR. Calculation remains per-group. If split_groups is False, this value has no effect.
            Set to False to have each group use its own learning rate. 
            (default: True)
        factored (boolean):
            Use factored approximation of the second moment, similar to Adafactor. Reduces memory usage. Disable
            if training results in NaNs or the learning rate fails to grow.
            (default: True)
        factored_fp32 (boolean):
            Force the use of float32 for the factored second moment. Because the factorisation is an approximation, it can
            be beneficial to use high precision to avoid stability issues. However, if you're training in lower precision 
            for short durations, setting this to False will slightly reduce memory usage. 
            Ignored if factored is False.
            (default: True)
        fused_back_pass (boolean):
            Stops the optimiser from running the normal step method. Set to True if using fused backward pass. Really only
            needed for scripts and UIs that call the regular step method even when using fused backward pass (OneTrainer).
            (default: False)
        use_stableadamw (boolean):
            Scales parameter updates by the root-mean-square of the normalised gradient, in essence identical to 
            Adafactor's gradient scaling. Set to False if the adaptive learning rate never improves.
            (default: True)
        use_muon_pp (boolean):
            Experimental. Perform orthogonalisation on the gradient before it is used for updates ala Shampoo/SOAP/Muon.
            (https://github.com/KellerJordan/Muon/blob/master/muon.py). Not suitable for all training scenarios.
            May not work well with small batch sizes or finetuning.
            (default: False)
        use_cautious (boolean):
            Experimental. Perform "cautious" updates, as proposed in https://arxiv.org/pdf/2411.16085. Modifies
            the update to isolate and boost values that align with the current gradient. Note that we do not have
            access to a first moment, so this deviates from the paper (we apply the mask directly to the update).
            May have a limited effect.
            (default: False)
        use_grams (boolean):
            Experimental. Perform "grams" updates, as proposed in https://arxiv.org/abs/2412.17107. Modifies 
            the update using sign operations that align with the current gradient. Note that we do not have
            access to a first moment, so this deviates from the paper (we apply the sign directly to the update).
            May have a limited effect.
            (default: False)            
        use_adopt (boolean):
            Experimental. Performs a modified step where the second moment is updated after the parameter update,
            so as not to include the current gradient in the denominator. This is a partial implementation of ADOPT 
            (https://arxiv.org/abs/2411.02853), as we don't have a first moment to use for the update.
            (default: False)
        use_orthograd (boolean):
            Experimental. Updates weights using the component of the gradient that is orthogonal to the current 
            weight direction, as described in "Grokking at the Edge of Numerical Stability" (https://arxiv.org/pdf/2501.04697).
            Can help prevent overfitting and improve generalisation.
            (default: False)
        use_focus (boolean):
            Experimental. Modifies the update step to better handle noise at large step sizes. From 
            "FOCUS: First-Order Concentrated Update Scheme" (https://arxiv.org/abs/2501.12243). This method is
            incompatible with factorisation, Muon and Adam-atan2.
            (default: False)
        stochastic_rounding (boolean):
            Use stochastic rounding for bfloat16 weights (https://github.com/pytorch/pytorch/issues/120376). Brings
            bfloat16 training performance close to that of float32.
            (default: True)
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.99), beta3=None,
                 weight_decay=0.0,
                 weight_decay_by_lr=True,
                 use_bias_correction=False,
                 d0=1e-6, d_coef=1.0,
                 prodigy_steps=0,
                 use_speed=False,
                 eps=1e-8,
                 split_groups=True,
                 split_groups_mean=True,
                 factored=True,
                 factored_fp32=True,
                 fused_back_pass=False,
                 use_stableadamw=True,
                 use_muon_pp=False,
                 use_cautious=False,
                 use_grams=False,
                 use_adopt=False,
                 use_orthograd=False,
                 use_focus=False,
                 stochastic_rounding=True):

        super().__init__(params=params, lr=lr, betas=betas, beta3=beta3,
                         weight_decay=weight_decay, weight_decay_by_lr=weight_decay_by_lr,
                         use_bias_correction=use_bias_correction,
                         d0=d0, d_coef=d_coef, prodigy_steps=prodigy_steps, use_speed=use_speed,
                         eps=eps, split_groups=split_groups,
                         split_groups_mean=split_groups_mean, factored=factored, factored_fp32=factored_fp32,
                         fused_back_pass=fused_back_pass, use_stableadamw=use_stableadamw,
                         use_muon_pp=use_muon_pp, use_cautious=use_cautious, use_grams=use_grams, 
                         use_adopt=use_adopt, use_orthograd=use_orthograd, use_focus=use_focus, 
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
    def initialise_state(self, p, group):
        state, needs_init = self.initialise_state_internal(p, group)

        if needs_init:
            state['z'] = p.detach().clone(memory_format=torch.preserve_format)
        
        return state

    @torch.no_grad()
    def update_params(self, y, z, update, group, dlr):
        beta1, _ = group['betas']
        decay = group['weight_decay']

        weight = dlr ** 2
        weight_sum = group['weight_sum'] + weight
        ckp1 = weight / weight_sum if weight_sum else 0

        xy_step = 1 - beta1 * (1 - ckp1)

        if decay != 0:
            # Weight decay at Y.
            if group['weight_decay_by_lr']:
                decay *= dlr

            y.sub_(y, alpha=decay * xy_step)
            z.sub_(y, alpha=decay)

        cautious, grams = group['use_cautious'], group['use_grams']

        if cautious or grams:
            u = (y - z).mul_(ckp1).add_(update, alpha=dlr * xy_step)
            z.sub_(update, alpha=dlr)

            if cautious:
                # "Cautious Optimizer (C-Optim): Improving Training with One Line of Code": https://github.com/kyleliang919/c-optim
                # ScheduleFree implementation by nhamanasu: https://github.com/facebookresearch/schedule_free/pull/54
                mask = update.mul_(u).sign_().clamp_min_(0)
                mask.mul_(mask.numel() / mask.sum().add(1))
                u.mul_(mask)
            elif grams:
                # "Grams: Gradient Descent with Adaptive Momentum Scaling": https://arxiv.org/abs/2412.17107
                u.abs_().mul_(update.sign_())

            y.sub_(u)
            del u
        else:
            y.lerp_(end=z, weight=ckp1)
            y.sub_(update, alpha=dlr * xy_step)
            z.sub_(update, alpha=dlr)

        return weight_sum

    @torch.no_grad()
    def step_param(self, p, group):
        self.on_start_step(p, group)

        if not group['train_mode']:
            raise Exception("Not in train mode!")

        weight_sum = group['weight_sum']

        if p.grad is not None:
            use_adopt = group['use_adopt']
            stochastic = group['stochastic_rounding']
            _, beta2 = group['betas']
            k = group['k']

            state = self.initialise_state(p, group)

            z_state = state['z']
            y, z = (p.float(), z_state.float()) if stochastic else (p, z_state)

            grad = self.orthograd(z_state, p.grad) if group['use_orthograd'] else p.grad.to(dtype=torch.float32, copy=True)
            dlr = self.get_dlr(group)

            if group['use_bias_correction']:
                beta2_t = beta2 ** k
                bias_correction2 = 1 - beta2_t

                # maximum length of the approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                # compute the length of the approximated SMA
                rho_t = rho_inf - 2 * k * beta2_t / bias_correction2
                rect = (
                    ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    if rho_t > 4.0
                    else 0.0
                )
                dlr *= rect
                beta2 = 1 - (1 - beta2) / (1 - beta2_t)

            update = None

            if state['muon']:
                grad = self.newton_schulz_(grad)
                if group['use_speed']:
                    grad_rms = state['rms_sq']
                    if grad_rms is None:
                        grad_rms = state['rms_sq'] = 1 / self.get_rms(grad)
                    update = grad.mul_(grad_rms)
                else:
                    d_k = group['d_prev'] / group['d']
                    rms_sq = state["rms_sq"].mul_(beta2 * d_k * d_k).add_(self.get_rms(grad).square(), alpha=1 - beta2)
                    update = grad.mul_(1 / rms_sq.sqrt().add(1e-8))
            else:
                if use_adopt and group['k'] == 1:
                    self.update_second_moment(state, group, grad, 0, y, return_denom=False)
                else:
                    denom = self.update_second_moment(state, group, grad, beta2, y, denom_before_update=use_adopt)
                    if group['use_bias_correction'] and rho_t <= 4.0:
                        update = grad
                    else:
                        update = self.update_(grad, denom, group, y)
                    del denom

            if update is not None:
                if group['use_stableadamw']:
                    clip_threshold = self.get_clip_threshold(group)
                    rms = self.get_rms(update, 1).div(clip_threshold).clamp_min(1)
                    update.mul_(1 / rms)

                self.update_prodigy(state, group, p.grad, p)

                weight_sum = self.update_params(y, z, update, group, dlr)

                self.smart_copy(p, y, stochastic, True)
                self.smart_copy(z_state, z, stochastic, True)

                del update

        group['running_weight_sum'] = weight_sum
        self.on_end_step()