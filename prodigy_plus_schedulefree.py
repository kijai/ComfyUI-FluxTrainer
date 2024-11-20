# source: @LoganBooker - https://github.com/LoganBooker/prodigy-plus-schedule-free
import math
import torch
import torch.optim
from statistics import mean, harmonic_mean, geometric_mean

class ProdigyPlusScheduleFree(torch.optim.Optimizer):
    r"""
    An optimiser based on Prodigy that includes schedule-free logic. Has additional improvements in the form of
    StableAdamW gradient scaling, per parameter group adaptation, lower memory utilisation and moving average stepsizes.

    Based on code from:
    https://github.com/facebookresearch/schedule_free
    https://github.com/konstmish/prodigy

    Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
    https://github.com/konstmish/prodigy/pull/23
    https://github.com/konstmish/prodigy/pull/22
    https://github.com/konstmish/prodigy/pull/20

    As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
    calls to train() and eval(). See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free
    
    If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic, 
    and you should set beta4 to 0 and prodigy_steps to None.

    Leave LR set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
    ability for the optimiser to predict stepsizes. Scaling of large gradients is already handled by StableAdamW, which
    effectively uses Adafactor's gradient clipping.

    For default Prodigy + schedule-free behaviour, set beta4 to 0 and prodigy_steps to None. Setting beta4 to None or a positive
    value will treat the stepsize as a running average, and allow the stepsize to both decrease and increase over time. This is
    contrary to Prodigy's default behaviour, which never decreases the stepsize.

    Recommended values for beta4 if set manually are 0.99-0.999, with lower values making the adaptation more noisy and aggressive.
    If beta4 is set to None, beta2 is used.
    
    By default, split_groups is set to True, so each parameter group will have its own adaptation values. So if you're training
    different networks together, they won't contaminate each other's learning rates. The disadvantage of this approach is that some 
    networks can take a long time to reach a good learning rate when trained alongside others (for example, SDXL's Unet). 
    It's recommended to use a higher d0 (1e-5, 5e-5, 1e-4) so these networks don't get stuck at a low learning rate.
    
    For Prodigy's default behaviour, which lumps all parameter groups together, set split_groups to False.

    In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
    can be controlled via the prodigy_steps settings.

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
            Adam-atan2 is used instead. This removes the need for epsilon tuning, but may not work well with newer diffusion models.
            (default: 1e-8).
        scale_atan2 (boolean):
            Ignored if eps is not None. Scale Adam-atan2 updates to more closely mimic division + epsilon updates. Set to True if
            Adam-atan2 updates fail to increase the learning rate.
            (default False)
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
            Use factored approximation of the second moment, similar to Adafactor. Reduces memory usage.
            (default False)
        fused_back_pass (boolean):
            Stops the optimiser from running the normal step method. Set to True if using fused backward pass.
            (default False)
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
                 factored=False,
                 fused_back_pass=False,
                 scale_atan2=False):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps is not None and not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if beta3 is not None and not 0.0 <= beta3 < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(beta3))
        if beta4 is not None and not 0.0 <= beta4 < 1.0:
            raise ValueError("Invalid beta4 parameter: {}".format(beta4))
        if split_groups_mean not in {None, "mean", "harmonic_mean", "geometric_mean"}:
            raise ValueError(f"Invalid value for split_groups_mean: '{split_groups_mean}'. Must be one of {None, 'mean', 'harmonic_mean', 'geometric_mean'}")

        defaults = dict(lr=lr, betas=betas, beta3=beta3, beta4=beta4,
                        eps=eps,
                        weight_decay=weight_decay,
                        d=d0, d0=d0, d_coef=d_coef,
                        k=1,initialised=None,
                        train_mode=True,
                        weight_sum=0,
                        prodigy_steps=prodigy_steps,
                        warmup_steps=warmup_steps,
                        lr_max=-1,
                        use_bias_correction=use_bias_correction,
                        d_numerator=0.0,
                        factored=factored,
                        scale_atan2=scale_atan2)

        super().__init__(params, defaults)

        self.d0 = d0
        if split_groups and len(self.param_groups) == 1:
            print("[Prodigy+ScheduleFree] Optimiser contains single param_group -- 'split_groups' has been disabled.")
            split_groups = False

        self.split_groups = split_groups
        self.split_groups_mean = split_groups_mean

        # Properties for fused backward pass.
        self.groups_to_process = None
        self.shared_d = None
        self.fused_back_pass = fused_back_pass

        self.running_d_numerator = None
        self.running_d_denom = None

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(device=p.device), weight=1 - 1 / beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(device=p.device), weight=1 - beta1)
                group['train_mode'] = True

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True
    
    def supports_fused_back_pass(self):
        return True

    @torch.no_grad()
    def get_sliced_tensor(self, tensor, slice_p=11):
        return tensor.ravel()[::slice_p]
   
    @torch.no_grad()
    def get_d_mean(self, groups, mode):
        if mode is None:
            return None
        elif mode == "harmonic_mean":
            return harmonic_mean(group['d'] for group in groups)
        elif mode == "geometric_mean":
            return geometric_mean(group['d'] for group in groups)
        elif mode == "mean":
            return mean(group['d'] for group in groups)
        
        raise ValueError(f"Invalid value for split_groups_mean: '{mode}'. Must be one of {None, 'mean', 'harmonic_mean', 'geometric_mean'}")

    # Modified Adafactor factorisation implementation by Ross Wightman 
    # https://github.com/huggingface/pytorch-image-models/pull/2320
    @torch.no_grad()
    def factored_dims(self,
        shape,
        factored,
        min_dim_size_to_factor):
        r"""Whether to use a factored second moment estimator.
        This function returns a tuple with the two largest axes to reduce over.
        If all dimensions have size < min_dim_size_to_factor, return None.
        Args:
        shape: an input shape
        factored: whether to use factored second-moment estimator for > 2d vars.
        min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
        Returns:
        None or a tuple of ints
        """
        if not factored or len(shape) < 2:
            return None
        if all(dim < min_dim_size_to_factor for dim in shape):
            return None
        sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
        return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])

    @torch.no_grad()
    def denom_from_state(self, exp_avg_sq):
        # Implicit detection of factored mode and single dim tensors.
        if isinstance(exp_avg_sq, list):
            row_var, col_var, _, _, reduce_dc = exp_avg_sq
            row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True)
            row_factor = row_var.div(row_col_mean).sqrt_()
            col_factor = col_var.sqrt()
            return row_factor * col_factor

        return exp_avg_sq.sqrt()
    
    @torch.no_grad()
    def initialise_state(self, p, state, factored, bf16_state=True):
        if p.grad is None or len(state) != 0:
            return

        grad = p.grad
        dtype = torch.bfloat16 if bf16_state and p.dtype is torch.float32 else p.dtype
        sliced_data = self.get_sliced_tensor(p)

        state['z'] = p.detach().clone(memory_format=torch.preserve_format)

        factored_dims = self.factored_dims(
            grad.shape,
            factored=factored,
            min_dim_size_to_factor=32
        )

        if factored_dims is not None:
            dc, dr = factored_dims
            row_shape = list(p.grad.shape)
            row_shape[dr] = 1
            col_shape = list(p.grad.shape)
            col_shape[dc] = 1
            reduce_dc = dc - 1 if dc > dr else dc
            # Store reduction variables so we don't have to recalculate each step.
            # Always store second moment low ranks in fp32 to avoid precision issues. Memory difference 
            # between bf16/fp16 and fp32 is negligible here.
            state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=torch.float32, device=p.device).detach(), 
                                   torch.zeros(col_shape, dtype=torch.float32, device=p.device).detach(), 
                                   dr, dc, reduce_dc]
        else:
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format).detach()
        
        # If the initial weights are zero, don't bother storing them.
        if p.count_nonzero() > 0:
            state['p0'] = sliced_data.to(dtype=dtype, memory_format=torch.preserve_format, copy=True).detach()
        else:
            state['p0'] = torch.tensor(0.0, dtype=dtype, device=p.device)
        
        state['s'] = torch.zeros_like(sliced_data, memory_format=torch.preserve_format, dtype=dtype).detach()

    @torch.no_grad()
    def update_d_and_reset(self, group):
        k = group['k']
        beta1, beta2 = group['betas']
        beta3, beta4 = group['beta3'], group['beta4']
        
        if beta3 is None:
            beta3 = beta2 ** 0.5

        if beta4 is None:
            beta4 = beta1 ** 0.5

        d = group['d']
        d0 = group['d0']
        d_coef = group['d_coef']
        prodigy_steps = group['prodigy_steps']

        d_numerator = group['d_numerator']
        d_numerator *= beta3

        d_numerator_item = self.running_d_numerator.item()
        d_denom_item = self.running_d_denom.item()

        # Prevent the accumulation of negative values in the numerator in early training.
        # We still allow negative updates once progress starts being made, as this is 
        # important for regulating the adaptive stepsize.
        if d_numerator_item > 0 or d > d0:
            d_numerator = max(0, d_numerator + d_numerator_item)

        if d_denom_item > 0 and (prodigy_steps <= 0 or k < prodigy_steps):
            d_hat = max(math.atan2(d_coef * d_numerator, d_denom_item), d)
            d = d * beta4 + d_hat * (1 - beta4) if beta4 > 0 else d_hat
        
        group['d'] = d
        group['d_numerator'] = d_numerator

        self.running_d_numerator.zero_()
        self.running_d_denom.zero_()

    @torch.no_grad()
    def step_param(self, p, group):
        if not group['train_mode']:
            raise Exception("Not in train mode!")

        if self.groups_to_process is None:
            # Optimiser hasn't run yet, so initialise.
            self.groups_to_process = {i: len(group['params']) for i, group in enumerate(self.param_groups)}

            # Use tensors to keep everything on device during parameter loop.
            self.running_d_numerator = torch.tensor(0.0, dtype=torch.float32, device=p.device)
            self.running_d_denom = torch.tensor(0.0, dtype=torch.float32, device=p.device)
        elif len(self.groups_to_process) == 0:
            # Start of new optimiser run, so grab updated d.
            self.groups_to_process = {i: len(group['params']) for i, group in enumerate(self.param_groups)}

            if not self.split_groups:
                # When groups aren't split, calculate d for the first group,
                # then copy to all other groups.
                self.update_d_and_reset(group)
                for g in self.param_groups:
                    g['d'] = group['d']

            self.shared_d = self.get_d_mean(self.param_groups, self.split_groups_mean) if self.split_groups else None

        k = group['k']

        group_index = self.param_groups.index(group)
        is_first_param_for_group = self.groups_to_process[group_index] == len(group['params'])

        if p.grad is not None:
            lr = group['lr']

            beta1, beta2 = group['betas']
            beta3 = group['beta3']
            eps = group['eps']

            warmup_steps = group['warmup_steps']

            d = group['d']
            d0 = group['d0']

            factored = group['factored']

            if beta3 is None:
                beta3 = beta2 ** 0.5

            dlr = (self.shared_d if self.split_groups and self.shared_d else d) * lr
            d_update = (d / d0) * dlr

            # Apply warmup separate to the denom and numerator updates.
            if k < warmup_steps:
                dlr *= k / warmup_steps

            # Adafactor / PaLM beta2 decay. Clip beta2 as per Scaling ViT paper.
            if group['use_bias_correction']:
                beta2 = min(1 - k ** -0.8, beta2)

            state = self.state[p]
            self.initialise_state(p, state, factored)

            one_minus_beta2_d = d * d * (1 - beta2)

            grad = p.grad.float()
            y, z, s = p, state['z'], state['s']

            sliced_grad = self.get_sliced_tensor(grad)
            sliced_data = self.get_sliced_tensor(z)
                
            exp_avg_sq = state['exp_avg_sq']

            # Adam EMA updates
            if isinstance(exp_avg_sq, list):
                grad_sqr = grad.square().add_(1e-30)
                row_var, col_var, dr, dc, _ = exp_avg_sq
                row_var.mul_(beta2).add_(grad_sqr.mean(dim=dr, keepdim=True), alpha=one_minus_beta2_d)
                col_var.mul_(beta2).add_(grad_sqr.mean(dim=dc, keepdim=True), alpha=one_minus_beta2_d)
                del grad_sqr
            else:
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2_d)

            x0_minus = state['p0'].float() - sliced_data.float()
            self.running_d_numerator.add_(torch.dot(sliced_grad, x0_minus), alpha=d_update)
            del x0_minus
            
            s.mul_(beta3).add_(sliced_grad, alpha=d_update)
            self.running_d_denom.add_(s.abs().sum())

            if is_first_param_for_group:
                lr_max = group['lr_max'] = max(dlr, group['lr_max'])
                weight = lr_max ** 2
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            else:
                lr_max = group['lr_max']
                weight = lr_max ** 2
                weight_sum = group['weight_sum']

            ckp1 = weight / weight_sum if weight_sum else 0

            weight_decay = group['weight_decay']

            denom = self.denom_from_state(exp_avg_sq)
            if eps is None:
                update = grad.mul_(d)

                # Adam-atan2. Use atan2 rather than epsilon and division 
                # for parameter updates (https://arxiv.org/abs/2407.05872).
                # Has the nice property of "clipping" the gradient as well.
                if group['scale_atan2']:
                    atan_scale = 1 / d
                    update.atan2_(denom.mul_(atan_scale)).mul_(atan_scale)
                else:
                    update.atan2_(denom)
            else:
                update = grad.div_(denom.add_(d * eps)).mul_(d)

            # Weight decay.
            update.add_(y, alpha=weight_decay)

            y.lerp_(end=z, weight=ckp1)
            y.add_(update, alpha=dlr * (beta1 * (1 - ckp1) - 1))

            z.sub_(update, alpha=dlr)
            del update, denom

        # Decrement params processed so far.
        self.groups_to_process[group_index] -= 1

        # End of param loop for group, update calculations.
        if self.groups_to_process[group_index] == 0:
            group['k'] = k + 1
            self.groups_to_process.pop(group_index)
            if self.split_groups:
                # When groups are split, calculate per-group d.
                self.update_d_and_reset(group)

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        self.step_param(p, group)

    @torch.no_grad()
    def step(self, closure=None):
        if self.fused_back_pass:
            return
        
        """Performs a single optimisation step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            for p in param_group["params"]:
                self.step_param(p, param_group)

        return loss