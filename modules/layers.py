# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math  # noqa: I001
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        assert flash_attn_varlen_func is not None
        x: torch.Tensor = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )  # type: ignore
        # x with shape [(bxs), a, d]
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # type: ignore # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert attn_mask is None, (
                "Causal mask and attn_mask cannot be used together"
            )
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def apply_gate(x, gate=None, tanh=False):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], device=device, dtype=dtype
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_channels, device=device, dtype=dtype)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], device=device, dtype=dtype
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, **factory_kwargs
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.mlp[2].weight, std=0.02)  # type: ignore

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim (int): the dimension of the output.
            max_period (int): controls the minimum frequency of the embeddings.

        Returns:
            embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

        .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)  # type: ignore
        t_emb = self.mlp(t_freq)
        return t_emb


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.gradient_checkpointing = False 

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)


def rope(pos, dim: int, theta: int):
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def attention_after_rope(q, k, v, pe):
    q, k = apply_rope(q, k, pe)

    from .attention import attention

    x = attention(q, k, v, mode="flash")
    return x


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def apply_rope(xq, xk, freqs_cis):
    # 将 num_heads 和 seq_len 的维度交换回原函数的处理顺序
    xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    xk = xk.transpose(1, 2)

    # 将 head_dim 拆分为复数部分（实部和虚部）
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    # 应用旋转位置编码（复数乘法）
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    # 恢复张量形状并转置回目标维度顺序
    xq_out = xq_out.reshape(*xq.shape).type_as(xq).transpose(1, 2)
    xk_out = xk_out.reshape(*xk.shape).type_as(xk).transpose(1, 2)

    return xq_out, xk_out


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def scale_add_residual(
    x: torch.Tensor, scale: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    return x * scale + residual


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def layernorm_and_scale_shift(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.size(-1),)) * (scale + 1) + shift


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention_after_rope(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    @staticmethod
    def rms_norm_fast(x, weight, eps):
        return LigerRMSNormFunction.apply(
            x,
            weight,
            eps,
            0.0,
            "gemma",
            True,
        )

    @staticmethod
    def rms_norm(x, weight, eps):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x * rrms).to(dtype=x_dtype) * weight

    def forward(self, x: Tensor):
        return self.rms_norm_fast(x, self.scale.to(x.dtype), 1e-6)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=1)
        k = torch.cat((txt_k, img_k), dim=1)
        v = torch.cat((txt_v, img_v), dim=1)

        attn = attention_after_rope(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img_mlp = self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )
        img = scale_add_residual(img_mlp, img_mod2.gate, img)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt_mlp = self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        txt = scale_add_residual(txt_mlp, txt_mod2.gate, txt)
        return img, txt

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, img, txt, vec, pe, use_reentrant=False)
            # cpu offload checkpointing

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    cuda_inputs = to_cuda(inputs)
                    outputs = func(*cuda_inputs)
                    return to_cpu(outputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward), img, txt, vec, pe, use_reentrant=False
            )

        else:
            return self._forward(img, txt, vec, pe)

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention_after_rope(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return scale_add_residual(output, mod.gate, x)
    
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, x, vec, pe, use_reentrant=False)

            # cpu offload checkpointing

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    cuda_inputs = to_cuda(inputs)
                    outputs = func(*cuda_inputs)
                    return to_cpu(outputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward), x, vec, pe, use_reentrant=False
            )
        else:
            return self._forward(x, vec, pe)

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
