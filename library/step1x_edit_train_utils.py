import argparse
import math
import os
import numpy as np
import toml
import json
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, PartialState
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file

from library import strategy_base, train_util, step1x_utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .utils import setup_logging, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput

# region sample images


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
# endregion


# region train
def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
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


def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, h, w = latents.shape
    assert bsz > 0, "Batch size not large enough"
    num_timesteps = noise_scheduler.config.num_train_timesteps
    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        # Simple random sigma-based noise sampling
        if args.timestep_sampling == "sigmoid":
            sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            sigmas = torch.rand((bsz,), device=device)

        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale  # larger scale for more uniform sampling
        sigmas = sigmas.sigmoid()
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_timesteps
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
        indices = (u * num_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)

    # Broadcast sigmas to latent shape
    sigmas = sigmas.view(-1, 1, 1, 1)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma = args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
    else:
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


def apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas):
    weighting = None
    if args.model_prediction_type == "raw":
        pass
    elif args.model_prediction_type == "additive":
        # add the model_pred to the noisy_model_input
        model_pred = model_pred + noisy_model_input
    elif args.model_prediction_type == "sigma_scaled":
        # apply sigma scaling
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

    return model_pred, weighting


def add_step1x_edit_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--clip_l",
        type=str,
        help="path to clip_l (*.sft or *.safetensors), should be float16 / clip_lのパス（*.sftまたは*.safetensors）、float16が前提",
    )
    parser.add_argument("--ae", type=str, help="path to ae (*.sft or *.safetensors) / aeのパス（*.sftまたは*.safetensors）")
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="path to controlnet (*.sft or *.safetensors) / controlnetのパス（*.sftまたは*.safetensors）",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the DIT.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift"],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal, shift of sigmoid.",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"の場合のみ有効）。',
    )
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw", "additive", "sigma_scaled"],
        default="sigma_scaled",
        help="How to interpret and process the model prediction: "
        "raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."
        " / モデル予測の解釈と処理方法："
        "raw（そのまま使用）、additive（ノイズ入力に加算）、sigma_scaled（シグマスケーリングを適用）。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 3.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは3.0。",
    )





@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor



class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps)

        sigmas = timesteps / self.config.num_train_timesteps
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
