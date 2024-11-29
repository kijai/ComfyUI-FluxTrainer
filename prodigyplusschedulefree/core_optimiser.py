import math
import torch
from statistics import mean, harmonic_mean, geometric_mean

class CoreOptimiser(torch.optim.Optimizer):
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

        if use_adopt and use_muon_pp:
            print(f"[{self.__class__.__name__}] Muon and ADOPT cannot be used at the same time. Muon has been disabled.")
            use_muon_pp = False

        defaults = dict(lr=lr, betas=betas, beta3=beta3, beta4=beta4,
                        eps=eps,
                        weight_decay=weight_decay,
                        d=d0, d0=d0, d_coef=d_coef,
                        k=1, train_mode=True,
                        weight_sum=0,
                        prodigy_steps=prodigy_steps,
                        warmup_steps=warmup_steps,
                        use_bias_correction=use_bias_correction,
                        d_numerator=0.0,
                        factored=factored,
                        use_stableadamw=use_stableadamw,
                        use_muon_pp=use_muon_pp,
                        use_cautious=use_cautious,
                        use_adopt=use_adopt,
                        stochastic_rounding=stochastic_rounding)

        super().__init__(params, defaults)

        self.d0 = d0
        if split_groups and len(self.param_groups) == 1:
            print(f"[{self.__class__.__name__}] Optimiser contains single param_group -- 'split_groups' has been disabled.")
            split_groups = False

        self.split_groups = split_groups
        self.split_groups_mean = split_groups_mean

        # Properties for fused backward pass.
        self.groups_to_process = None
        self.shared_d = None
        self.fused_back_pass = fused_back_pass

        # Use tensors to keep everything on device during parameter loop.
        for group in (self.param_groups if self.split_groups else self.param_groups[:1]):
            p = group['params'][0]
            group['running_d_numerator'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
            group['running_d_denom'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def eval(self):
        pass

    @torch.no_grad()
    def train(self):
        pass

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
    def get_running_values_for_group(self, group):
        if not self.split_groups:
            group = self.param_groups[0]
        return group['running_d_numerator'], group['running_d_denom']

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

    # From: https://github.com/KellerJordan/Muon/blob/master/muon.py
    @torch.no_grad()
    def newton_schulz_(self, G, steps=6, eps=1e-7):
        # Inline reshaping step within the method itself.
        original_shape = None
        if len(G.shape) > 2:
            original_shape = G.shape
            G = G.view(G.size(0), -1)
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps) # ensure top singular value <= 1
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        if X is not G:
            G.copy_(X)
            del X
        if original_shape is not None:
            G = G.view(*original_shape)
        return G
    
    # Implementation by Nerogar. From: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    def copy_stochastic_(self, target, source):
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

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
    def initialise_state(self, p, factored, use_muon_pp):
        raise Exception("Not implemented!")

    @torch.no_grad()
    def initialise_state_internal(self, p, factored, use_muon_pp):
        state = self.state[p]
        needs_init = len(state) == 0
        
        if needs_init:
            grad = p.grad
            dtype = torch.bfloat16 if p.dtype == torch.float32 else p.dtype
            sliced_data = self.get_sliced_tensor(p)

            # NOTE: We don't initialise z/exp_avg here -- subclass needs to do that.
            state['muon'] = use_muon_pp and len(grad.shape) >= 2 and grad.size(0) < 10000

            if not state['muon']:
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
        
        return state, needs_init

    @torch.no_grad()
    def update_d_and_reset(self, group):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps > 0 and k >= prodigy_steps:
            return

        beta1, beta2 = group['betas']
        beta3, beta4 = group['beta3'], group['beta4']
        
        if beta3 is None:
            beta3 = beta2 ** 0.5

        if beta4 is None:
            beta4 = beta1 ** 0.5

        d = group['d']
        d0 = group['d0']
        d_coef = group['d_coef']

        running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

        d_numerator = group['d_numerator']
        d_numerator *= beta3

        d_numerator_item = running_d_numerator.item()
        d_denom_item = running_d_denom.item()

        # Prevent the accumulation of negative values in the numerator in early training.
        # We still allow negative updates once progress starts being made, as this is 
        # important for regulating the adaptive stepsize.
        if d_numerator_item > 0 or d > d0:
            d_numerator = max(0, d_numerator + d_numerator_item)

        if d_denom_item > 0:
            d_hat = max(math.atan2(d_coef * d_numerator, d_denom_item), d)
            d = d * beta4 + d_hat * (1 - beta4) if beta4 > 0 else d_hat
        
        group['d'] = d
        group['d_numerator'] = d_numerator

        running_d_numerator.zero_()
        running_d_denom.zero_()

    def on_start_step(self, group):
        if self.groups_to_process is None:
            # Optimiser hasn't run yet, so initialise.
            self.groups_to_process = {i: len(group['params']) for i, group in enumerate(self.param_groups)}
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

    def on_end_step(self, group):
        group_index = self.param_groups.index(group)

        # Decrement params processed so far.
        self.groups_to_process[group_index] -= 1

        # End of param loop for group, update calculations.
        if self.groups_to_process[group_index] == 0:
            k = group['k']
            prodigy_steps = group['prodigy_steps']
            if prodigy_steps > 0 and k == prodigy_steps:
                print(f"[{self.__class__.__name__}] Prodigy stepsize adaptation disabled after {k} steps for param_group {group_index}.")

            self.groups_to_process.pop(group_index)
            if self.split_groups: # When groups are split, calculate per-group d.
                self.update_d_and_reset(group)

            group['k'] = k + 1
            return True

        return False

    def get_dlr(self, group):
        lr = group['lr']
        k = group['k']

        warmup_steps = group['warmup_steps']

        d = group['d']
        dlr = (self.shared_d if self.split_groups and self.shared_d else d) * lr

        # Apply warmup separate to the denom and numerator updates.
        if k < warmup_steps:
            dlr *= k / warmup_steps  
        
        return dlr

    def update_prodigy(self, state, group, grad, data, dlr):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps <= 0 or k < prodigy_steps:
            d, d0 = group['d'], group['d0']
            beta3 = group['beta3']

            if beta3 is None:
                beta3 = group['betas'][1] ** 0.5

            sliced_grad = self.get_sliced_tensor(grad)
            sliced_data = self.get_sliced_tensor(data)

            running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

            s = state['s']
            x0_minus = state['p0'] - sliced_data
            running_d_numerator.add_(torch.dot(sliced_grad, x0_minus), alpha=(d / d0) * dlr)
            del x0_minus
            
            s.mul_(beta3).add_(sliced_grad, alpha=(d / d0) * dlr)
            running_d_denom.add_(s.abs().sum())
        elif 's' in state: # Free the memory used by Prodigy, as we no longer need it.
            del state['s']
            del state['p0']

    def get_update(self, num, denom, group):
        d = group['d']

        if group['eps'] is None:
            # Adam-atan2. Use atan2 rather than epsilon and division 
            # for parameter updates (https://arxiv.org/abs/2407.05872).
            # Has the nice property of "clipping" the gradient as well.
            update = num.mul_(d).atan2_(denom)
        else:
            # Assume eps as already been added.
            update = num.div_(denom).mul_(d)

        return update

    def get_denom(self, state, group):
        exp_avg_sq = state['exp_avg_sq']
        eps = group['eps']

         # Adam EMA updates
        if isinstance(exp_avg_sq, list):
            row_var, col_var, _, _, reduce_dc = exp_avg_sq

            row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(1e-30)
            row_factor = row_var.div(row_col_mean).sqrt_()
            col_factor = col_var.sqrt()
            denom = row_factor * col_factor
        else:
            denom = exp_avg_sq.sqrt()
       
        if eps is not None:
            denom.add_(group['d'] * eps)

        return denom

    def update_first_moment(self, exp_avg, group, grad):
        d = group['d']
        beta1, _ = group['betas']
       
        exp_avg.mul_(beta1).add_(grad, value=d * (1 - beta1))
        return exp_avg
    
    def update_second_moment(self, state, group, grad, beta2, return_denom=True):
        d = group['d']
        exp_avg_sq = state['exp_avg_sq']

        # Adafactor / PaLM beta2 decay. Clip beta2 as per Scaling ViT paper.
        if group['use_bias_correction']:
            beta2 = min(1 - group['k'] ** -0.8, beta2)

        one_minus_beta2_d = d * d * (1 - beta2)
    
        # Adam EMA updates
        if isinstance(exp_avg_sq, list):
            row_var, col_var, dr, dc, _ = exp_avg_sq

            row_var.mul_(beta2).add_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]), 
            alpha=one_minus_beta2_d)
            col_var.mul_(beta2).add_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]), 
            alpha=one_minus_beta2_d)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2_d)

        return self.get_denom(state, group) if return_denom else None
        
    def rms_(self, tensor, rms_min):
        if rms_min is not None:
            rms = tensor.norm().div(tensor.numel() ** 0.5).add(rms_min)
            tensor.div_(rms)
        return tensor

    # "Cautious Optimizer (C-Optim): Improving Training with One Line of Code"
    # https://github.com/kyleliang919/c-optim
    def cautious_(self, update, grad, reuse_grad):
        if reuse_grad:
            mask = grad.mul_(update) > 0
        else:
            mask = grad.mul(update) > 0

        mask_scale = mask.numel() / mask.sum().add(1)
        update.mul_(mask).mul_(mask_scale)
        del mask

        return update

    @torch.no_grad()
    def step_param(self, p, group):
        raise Exception("Not implemented!")            

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