import math
from dataclasses import dataclass

import numpy as np
import torch

from library import custom_offloading_utils

from torch import Tensor, nn

from .connector_edit import Qwen2Connector
from .layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock


@dataclass
class Step1XParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool


class Step1XEdit(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: Step1XParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.connector = Qwen2Connector()

        # adapted from kohya definition
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.blocks_to_swap = None

        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        self.time_in.enable_gradient_checkpointing()
        self.vector_in.enable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)

        print(f"Base model: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        self.time_in.disable_gradient_checkpointing()
        self.vector_in.disable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("Base Model: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks
        double_blocks_to_swap = num_blocks // 2
        single_blocks_to_swap = (num_blocks - double_blocks_to_swap) * 2

        assert double_blocks_to_swap <= self.num_double_blocks - 2 and single_blocks_to_swap <= self.num_single_blocks - 2, (
            f"Cannot swap more than {self.num_double_blocks - 2} double blocks and {self.num_single_blocks - 2} single blocks. "
            f"Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks."
        )

        self.offloader_double = custom_offloading_utils.ModelOffloader(
            self.double_blocks, self.num_double_blocks, double_blocks_to_swap, device  # , debug=True
        )
        self.offloader_single = custom_offloading_utils.ModelOffloader(
            self.single_blocks, self.num_single_blocks, single_blocks_to_swap, device  # , debug=True
        )
        print(
            f"Base model: Block swap enabled. Swapping {num_blocks} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_double_blocks = self.double_blocks
            save_single_blocks = self.single_blocks
            self.double_blocks = None
            self.single_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = save_double_blocks
            self.single_blocks = save_single_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    @staticmethod
    def timestep_embedding(
        t: Tensor, dim, max_period=10000, time_factor: float = 1000.0
    ):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        llm_embedding: Tensor,
        t_vec: Tensor,
        mask: Tensor,
    ) -> Tensor:
        txt, y = self.connector(
            llm_embedding, t_vec, mask
        )
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if not self.blocks_to_swap:
            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            img = torch.cat((txt, img), 1)
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe)
        else:
            for block_idx, block in enumerate(self.double_blocks):
                self.offloader_double.wait_for_block(block_idx)
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                self.offloader_double.submit_move_blocks(self.double_blocks, block_idx)

            img = torch.cat((txt, img), 1)

            for block_idx, block in enumerate(self.single_blocks):
                self.offloader_single.wait_for_block(block_idx)
                img = block(img, vec=vec, pe=pe)
                self.offloader_single.submit_move_blocks(self.single_blocks, block_idx)
        img = img[:, txt.shape[1] :, ...]

        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


if __name__ == "__main__":
    # Example usage
    params = Step1XParams(
        in_channels=768,
        out_channels=768,
        vec_in_dim=256,
        context_in_dim=768,
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=12,
        depth=12,
        depth_single_blocks=6,
        axes_dim=[1, 2, 3],
        theta=10000,
        qkv_bias=True
    )
    model = Step1XEdit(params)