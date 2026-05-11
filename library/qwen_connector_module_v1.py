import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _load_weights_file(file: str) -> Dict[str, torch.Tensor]:
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        return load_file(file)
    return torch.load(file, map_location="cpu")


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }


class QwenConnectorNetwork(nn.Module):
    def __init__(self, multiplier: float = 1.0) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.connector_ref: Optional[list[nn.Module]] = None

    @property
    def connector(self) -> nn.Module:
        if self.connector_ref is None or len(self.connector_ref) == 0 or self.connector_ref[0] is None:
            raise RuntimeError("connector is not attached. call apply_to() before using the network")
        return self.connector_ref[0]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.connector_ref is not None and len(self.connector_ref) > 0 and self.connector_ref[0] is not None:
            self.connector_ref[0].train(mode)
        return self

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        if self.connector_ref is None:
            return iter(())
        return self.connector.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        if self.connector_ref is None:
            return iter(())
        return self.connector.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def apply_to(self, text_encoders, dit, apply_text_encoder: bool = True, apply_unet: bool = True):
        if not apply_unet:
            raise ValueError("QwenConnectorNetwork requires apply_unet=True because connector belongs to the base DiT")
        self.connector_ref = [dit.connector]
        logger.info("enable connector-only training for U-Net connector")

    def is_mergeable(self):
        return False

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        lr = unet_lr if unet_lr is not None else default_lr
        if lr is None or lr == 0:
            raise ValueError("unet_lr or learning_rate must be specified for connector training")
        params = list(self.connector.parameters())
        if len(params) == 0:
            raise ValueError("no connector parameters found")
        return [{"params": params, "lr": lr}], ["unet_connector"]

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        connector = self.connector
        for param in connector.parameters():
            if _is_fp8_dtype(param.dtype):
                raise ValueError("QwenConnectorNetwork v1 does not support fp8_base/fp8_base_unet")
        connector.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.connector.parameters()

    def save_weights(self, file, dtype, metadata):
        from library import train_util

        state_dict = self.connector.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)
        else:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().clone().to("cpu")

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            metadata = {} if metadata is None else dict(metadata)
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        weights_sd = _load_weights_file(file)
        info = self.connector.load_state_dict(weights_sd, strict=True)
        return info

    def merge_to(self, text_encoders, dit, weights_sd, dtype=None, device=None):
        self.apply_to(text_encoders, dit, apply_text_encoder=False, apply_unet=True)
        target_connector = self.connector
        if dtype is not None:
            converted = {}
            for key, value in weights_sd.items():
                converted[key] = value.to(device=device or value.device, dtype=dtype)
            weights_sd = converted
        target_connector.load_state_dict(weights_sd, strict=True)
        logger.info("connector weights are loaded into the base DiT")


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    ae: Any,
    text_encoders,
    base_dit,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    return QwenConnectorNetwork(multiplier=multiplier)


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    base_dit,
    weights_sd=None,
    for_inference=False,
    **kwargs,
) -> Tuple[QwenConnectorNetwork, Dict[str, torch.Tensor]]:
    if weights_sd is None:
        if file is None:
            raise ValueError("file must be specified when weights_sd is None")
        weights_sd = _load_weights_file(file)
    network = QwenConnectorNetwork(multiplier=multiplier)
    return network, weights_sd
