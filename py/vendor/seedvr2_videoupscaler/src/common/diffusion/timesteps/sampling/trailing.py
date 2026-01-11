# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch

from ...types import SamplingDirection
from ..base import SamplingTimesteps


class KarrasScheduler:
    def __init__(self, T: int, steps: int, sigma_max: float = 14.614,
                 sigma_min: float = 0.002, rho: float = 7.0, **kwargs):
        """
        Karras schedule for few-step generation.

        Args:
            steps: Number of sampling steps (use 3)
            sigma_max: Starting noise (14.614 for FLUX-like models)
            sigma_min: Ending noise (0.002 for clean results)
            rho: Schedule steepness (7.0 = Karras default, higher = faster drop)
        """
        # Generate Karras sigmas
        ramp = torch.linspace(1, 0, steps + 1)  # +1 for final 0
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Optional: Convert to timesteps if your model expects [0,T] range
        timesteps = sigmas * (T / sigma_max)  # Scale to your T range

        super().__init__(T=T, timesteps=timesteps, direction=SamplingDirection.backward)


class UniformTrailingSamplingTimesteps(SamplingTimesteps):
    """
    Uniform trailing sampling timesteps.
    Defined in (https://arxiv.org/abs/2305.08891)

    Shift is proposed in SD3 for RF schedule.
    Defined in (https://arxiv.org/pdf/2403.03206) eq.23
    """

    def __init__(
        self,
        T: int,
        steps: int,
        shift: float = 1.0,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        # Create trailing timesteps with specified dtype
        shift = 10 #TBG shift for lesser artefects on high steps
        timesteps = torch.arange(1.0, 0.0, -1.0 / steps, device='cpu').to(device=device, dtype=dtype)

        # Shift timesteps.
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)

        # Scale to T range.
        if isinstance(T, float):
            timesteps = timesteps * T
        else:
            timesteps = timesteps.mul(T + 1).sub(1).round().int()

        super().__init__(T=T, timesteps=timesteps, direction=SamplingDirection.backward)


from comfy.samplers import KSampler, calculate_sigmas


# In your inference code:
def configure_diffusion(self, device, dtype):
    # ... existing code ...

    # Get sampler name from config
    sampler_name = self.config.diffusion.sampler.name  # "dpmpp_2m", "euler", etc.
    scheduler = self.config.diffusion.schedule.name  # "karras", "normal", etc.

    # Let ComfyUI calculate sigmas (handles Karras automatically)
    self.sigmas = calculate_sigmas(
        self.model.get_model_object("model_sampling"),
        scheduler,  # "karras" for few steps
        self.steps
    ).to(device=device, dtype=dtype)
