import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from library import kohya_trainer
from library import (
    step1x_edit_train_utils,
    step1x_utils,
    strategy_step1x,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Step1XEditNetworkTrainer(kohya_trainer.NetworkTrainer):
    def __init__(self):
        """
        初始化 Step1XEditNetworkTrainer 类。
        """
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_schnell: Optional[bool] = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        """
        断言额外的参数是否有效。

        Args:
            args: 命令行参数。
            train_dataset_group: 训练数据集组。
            val_dataset_group: 验证数据集组。
        """
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if args.fp8_base_unet:
            args.fp8_base = True  # 如果启用了 fp8_base_unet，则 fp8_base 也为 base model 启用

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_disk 已启用，因此 cache_text_encoder_outputs 也将启用"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / 缓存文本编码器输出时，caption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rate 不能使用"

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in base model training / max_token_length 在基模训练中未使用")

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swap 与 cpu_offload_checkpointing 不兼容"

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO 检查这个
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)  # TODO 检查这个

    def load_target_model(self, args, weight_dtype, accelerator):
        """
        加载目标模型（base模型、文本编码器、AE）。

        Args:
            args: 命令行参数。
            weight_dtype: 权重的数据类型。
            accelerator: Accelerator 对象。

        Returns:
            Tuple: 包含模型版本、文本编码器列表、AE 模型和 base 模型的元组。
        """
        # 当前将某些模型卸载到 CPU

        # 如果文件是 fp8 并且我们正在使用 fp8_base，我们可以按原样加载它 (fp8)
        loading_dtype = None if args.fp8_base else weight_dtype

        model = step1x_utils.load_models(
            dit_path=args.pretrained_model_name_or_path,
            device='cpu',
            dtype=loading_dtype
        )
        if args.fp8_base:
            # 检查模型的 dtype
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 model")
            else:
                logger.info(
                    "Cast model to fp8. This may take a while. You can reduce the time by using fp8 checkpoint."
                    " / 正在将模型转换为 fp8。这可能需要一些时间。您可以使用 fp8 检查点来缩短时间。"
                )
                model.to(torch.float8_e4m3fn)

        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            # 在前向和后向传递中，在 CPU 和 GPU 之间交换块以减少内存使用。
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        # 如果文件是 fp8 并且我们正在使用 fp8_base (而不是 unet)，我们可以按原样加载它 (fp8)
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # 按原样
        else:
            loading_dtype = weight_dtype

        qwen2p5vl = step1x_utils.load_qwen2p5vl(
            args.qwen2p5vl, dtype=weight_dtype, device="cpu"
        )
        qwen2p5vl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            # 检查模型的 dtype
            if qwen2p5vl.dtype == torch.float8_e4m3fnuz or qwen2p5vl.dtype == torch.float8_e5m2 or qwen2p5vl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {qwen2p5vl.dtype}")
            elif qwen2p5vl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 qwen2p5vl model")

        ae = step1x_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

        return "Step1X-Edit", [qwen2p5vl], ae, model

    def get_tokenize_strategy(self, args):
        """
        获取分词策略。
        """
        return strategy_step1x.Step1xEditTokenizeStrategy(tokenizer_cache_dir=args.qwen2p5vl)

    def get_tokenizers(self, tokenize_strategy):
        return [tokenize_strategy.processor]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_step1x.Step1XEditLatentsCachingStrategy(
            cache_to_disk=args.cache_latents_to_disk,
            batch_size=args.vae_batch_size,
            skip_disk_cache_validity_check=False,            
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_step1x.Step1XEditEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass 

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """
        获取用于文本编码的模型。

        Args:
            args: 命令行参数。
            accelerator: Accelerator 对象。
            text_encoders: 文本编码器列表。

        Returns:
            Optional[List[torch.nn.Module]]: 用于文本编码的模型列表，如果不需要则返回 None。
        """
        if args.cache_text_encoder_outputs:
            return None  # 不需要文本编码器进行编码，因为两者都已缓存
        else:
            return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_step1x.Step1xEditEncoderOutputsCachingStrategy(
                cache_to_disk=args.cache_text_encoder_outputs_to_disk,
                batch_size=args.text_encoder_batch_size,
                skip_disk_cache_validity_check=args.skip_cache_check,
                is_partial=False,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        """
        如果需要，缓存文本编码器的输出。

        Args:
            args: 命令行参数。
            accelerator: Accelerator 对象。
            unet: U-Net 模型。
            vae: VAE 模型。
            text_encoders: 文本编码器列表。
            dataset: 数据集组。
            weight_dtype: 权重的数据类型。
        """
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # 减少内存消耗
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # 当 TE 未被训练时，它不会被准备，所以我们需要使用显式的 autocast
            logger.info("move text encoders to gpu")
            # text_encoders[0].to(accelerator.device, dtype=weight_dtype)  # 始终不是 fp8
            [text_encoder.to(accelerator.device) for text_encoder in text_encoders]

            if text_encoders[0].dtype == torch.float8_e4m3fn:
                # 如果我们加载 fp8 权重，模型已经是 fp8，所以我们按原样使用它
                self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
            else:
                # 否则，我们需要将其转换为目标 dtype
                text_encoders[0].to(weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # 缓存样本提示
            if args.sample_prompts is not None:
                raise ValueError('not converted')

            accelerator.wait_for_everyone()

            # 移回 CPU
            if not self.is_train_text_encoder(args):
                text_encoders[0].to("cpu")
            # text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # 每次都从文本编码器获取输出，因此将其放在 GPU 上
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            # text_encoders[1].to(accelerator.device)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        """
        获取噪声调度器。

        Args:
            args: 命令行参数。
            device: 设备 (CPU 或 GPU)。

        Returns:
            Any: 噪声调度器对象。
        """
        noise_scheduler = step1x_edit_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        """
        将图像编码为潜变量。

        Args:
            args: 命令行参数。
            vae: VAE 模型。
            images: 图像张量。

        Returns:
            torch.Tensor: 潜变量张量。
        """
        import pdb;pdb.set_trace()
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        """
        对潜变量进行移位和缩放。

        Args:
            args: 命令行参数。
            latents: 潜变量张量。

        Returns:
            torch.Tensor: 经过移位和缩放的潜变量张量。
        """
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        ref_latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        """
        获取噪声预测和目标。
        这里之所以有了batch还要有latents和text_encoder_conds是因为有可能没有采取cache策略
        这部分的处理是在外部完成的

        Args:
            args: 命令行参数。
            accelerator: Accelerator 对象。
            noise_scheduler: 噪声调度器。
            latents: 潜变量。
            batch: 当前批次的数据。
            text_encoder_conds: 文本编码器的条件。
            unet: 基模
            network: 训练的网络。
            weight_dtype: 权重的数据类型。
            train_unet: 是否训练 U-Net。
            is_train: 是否处于训练模式。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                模型预测、目标、时间步长和权重 (如果适用)。
        """
        # 采样我们将添加到潜变量中的噪声
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # 获取带噪模型输入和时间步长
        noisy_model_input, timesteps, sigmas = step1x_edit_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # 打包潜变量并获取 img_ids
        packed_noisy_model_input = step1x_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = step1x_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # 处理ref_latents
        packed_ref_model_input = step1x_utils.pack_latents(ref_latents)

        # concate latents
        packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_ref_model_input], dim=1)
        img_ids = torch.cat([img_ids, img_ids], dim=1)

        # 确保隐藏状态需要梯度
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)
            img_ids.requires_grad_(True)

        # 预测噪声残差
        embeds, masks = text_encoder_conds
        masks = masks.to(torch.long)
        txt_ids = torch.zeros(bsz, embeds.shape[1], 3).to(packed_noisy_model_input.device)
        with torch.set_grad_enabled(is_train), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 原版有个标记，YiYi 注意：暂时将其除以 1000，因为我们在 Transformer 模型中将其缩放了 1000（我们不应该保留它，但我想保持模型的输入相同以进行测试）
            packed_noisy_model_input = packed_noisy_model_input.to(weight_dtype)
            masks = masks.to(device=accelerator.device)
            model_pred = unet(
                img=packed_noisy_model_input,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=timesteps / 1000,
                llm_embedding=embeds,
                t_vec=timesteps,
                mask=masks,
            )

        def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
            """
            x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
            """
            import einops
            x = einops.rearrange(x, "b (p h w) (c ph pw) -> b p c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2, p=2)
            return x[:, 0]

        # 解包潜变量
        model_pred = unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        weighting = None

        # 流匹配损失：这与 SD3 不同
        target = noise - latents

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        """
        后处理损失。

        Args:
            loss: 计算得到的损失。
            args: 命令行参数。
            timesteps: 时间步长。
            noise_scheduler: 噪声调度器。

        Returns:
            torch.Tensor: 后处理后的损失。
        """
        return loss

    def get_sai_model_spec(self, args):
        """
        获取 SAI 模型规范。

        Args:
            args: 命令行参数。

        Returns:
            Dict: SAI 模型规范字典。
        """
        return {
            # === Must ===
            "modelspec.sai_model_spec": "1.0.0",  # Required version ID for the spec
            "modelspec.architecture": "Step1X-Edit",
            "modelspec.implementation": "https://github.com/stepfun-ai/Step1X-Edit",
            "modelspec.title": "Lora",
            "modelspec.resolution": "1024",
            # === Should ===
            "modelspec.description": "Lora for Step1X-Edit",
            "modelspec.author": "Step1X-Edit Team",
            "modelspec.date": "2025",
        }

    def update_metadata(self, metadata, args):
        """
        更新元数据。

        Args:
            metadata: 要更新的元数据字典。
            args: 命令行参数。
        """
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        """
        判断在训练过程中是否不需要文本编码器。

        Args:
            args: 命令行参数。

        Returns:
            bool: 如果不需要文本编码器则返回 True，否则返回 False。
        """
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """
        为文本编码器的梯度检查点准备解决方法。
        """
        if index == 0:
            text_encoder.model.model.embed_tokens.encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        """
        为 fp8 准备文本编码器。

        Args:
            index: 文本编码器的索引 
            text_encoder: 文本编码器模型。
            te_weight_dtype: 文本编码器的权重数据类型。
            weight_dtype: 整体权重的数据类型。
        """
        raise ValueError('qwen still not tested for fp8')
        if step1x_utils.get_qwen_actual_dtype(text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
            logger.info(f"Qwen already prepared for fp8")
        else:
            logger.info(f"prepare Qwen for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
            text_encoder.to(te_weight_dtype)  # fp8

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        """
        在验证步骤结束时调用。

        Args:
            args: 命令行参数。
            accelerator: Accelerator 对象。
            network: 网络模型。
            text_encoders: 文本编码器列表。
            unet: U-Net 模型。
            batch: 当前批次的数据。
            weight_dtype: 权重的数据类型。
        """
        if self.is_swapping_blocks:
            # 为下一次前向传播做准备：因为没有调用后向传播，所以我们需要在这里准备
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        """
        使用 Accelerator 准备 U-Net 模型。

        Args:
            args: 命令行参数。
            accelerator: Accelerator 对象。
            unet: U-Net 模型。

        Returns:
            torch.nn.Module: 准备好的 U-Net 模型。
        """
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        # 如果我们不交换块，我们可以将模型移动到设备
        new_unet = unet
        new_unet = accelerator.prepare(new_unet, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(new_unet).move_to_device_except_swap_blocks(accelerator.device)  # 减少峰值内存使用
        accelerator.unwrap_model(new_unet).prepare_block_swap_before_forward()

        return new_unet


def setup_parser() -> argparse.ArgumentParser:
    """
    设置命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器对象。
    """
    parser = kohya_trainer.setup_parser()
    train_util.add_dit_training_arguments(parser)
    step1x_edit_train_utils.add_step1x_edit_train_arguments(parser)
    parser.add_argument('--qwen2p5vl', type=str, help='Path to Qwen2.5VL model / Qwen2.5VL模型的路径')
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = Step1XEditNetworkTrainer()
    trainer.train(args)