import os
import glob
from typing import Any, List, Optional, Tuple, Union
import PIL 
import torch
import numpy as np
from transformers import AutoProcessor

from library import train_util, step1x_utils
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from qwen_vl_utils import process_vision_info

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

Qwen25VL_7b_PREFIX = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''

def split_string(s):
    s = s.replace("“", '"').replace("”", '"').replace("'", '''"''')  # use english quotes
    result = []
    in_quotes = False
    temp = ""

    for idx,char in enumerate(s):
        if char == '"' and idx>155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""

            in_quotes = not in_quotes
            continue
        if in_quotes:
            if char.isspace():
                pass  # have space token

            result.append("“" + char + "”")
        else:
            temp += char

    if temp:
        result.append(temp)

    return result

class Step1xEditTokenizeStrategy(TokenizeStrategy):
    def __init__(self, max_length: int = 640, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.max_length = max_length 
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_cache_dir, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28
        )
        self.prefix = Qwen25VL_7b_PREFIX

    def tokenize(self, text: Union[str, List[str]], ref_images: Union[PIL.Image.Image, List[PIL.Image.Image]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        ref_images = [ref_images] if isinstance(ref_images, PIL.Image.Image) else ref_images

        res_list = []

        for idx, (txt, imgs) in enumerate(zip(text, ref_images)):
            messages = [{"role": "user", "content": []}]

            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})

            messages[0]["content"].append({"type": "image", "image": imgs})

            # 再添加 text
            messages[0]["content"].append({"type": "text", "text": f"{txt}"})

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            old_inputs_ids = inputs.input_ids
            text_split_list = split_string(text)

            token_list = []
            for text_each in text_split_list:
                txt_inputs = self.processor(
                    text=text_each,
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                token_each = txt_inputs.input_ids
                if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                    token_each = token_each[:, 1:-1]
                    token_list.append(token_each)
                else:
                    token_list.append(token_each)

            new_txt_ids = torch.cat(token_list, dim=1).to("cuda")

            new_txt_ids = new_txt_ids.to(old_inputs_ids.device)

            idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
            idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
            inputs.input_ids = (
                torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0)
                .unsqueeze(0)
                .to("cuda")
            )
            inputs.attention_mask = (inputs.input_ids > 0).long().to("cuda")

            res_list.append(inputs)

        return res_list

class Step1XEditEncodingStrategy(TextEncodingStrategy):
    def __init__(self, max_length=640, hidden_size=None) -> None:
        self.max_length = max_length
        self.hidden_size=hidden_size
        self.dtype = None

    def encode_tokens(self, tokenize_strategy, models, tokens):
        qwen2p5vl = models[0]
        if self.dtype is None:
            self.dtype = qwen2p5vl.model.lm_head.weight.dtype
        if self.hidden_size is None:
            self.hidden_size = qwen2p5vl.model.config.hidden_size
        embs = torch.zeros(
            len(tokens),
            self.max_length,
            self.hidden_size,
            dtype=self.dtype,
            device=torch.cuda.current_device()
        )
        masks = torch.zeros(
            len(tokens),
            self.max_length,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        self.device = torch.device(torch.cuda.current_device())
        for idx, inputs in enumerate(tokens):
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(self.dtype)
            # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = qwen2p5vl.model(
                input_ids = inputs.input_ids.to(torch.cuda.current_device()),
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values.to(torch.cuda.current_device()),
                image_grid_thw=inputs.image_grid_thw.to(torch.cuda.current_device()),
                output_hidden_states=True,
            )
            emb = outputs['hidden_states'][-1]
            embs[idx, : min(self.max_length, emb.shape[1] - 217)] = emb[0, 217:][
                : self.max_length
            ]

            masks[idx, : min(self.max_length, emb.shape[1] - 217)] = torch.ones(
                (min(self.max_length, emb.shape[1] - 217)),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )
        return embs, masks

class Step1xEditEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    Step1XEdit_ENCODER_OUTPUTS_NPZ_SUFFIX = "_step1x_te.npz"
    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

        self.warn_fp8_weights = False

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + Step1xEditEncoderOutputsCachingStrategy.Step1XEdit_ENCODER_OUTPUTS_NPZ_SUFFIX
    
    def is_disk_cached_outputs_expected(self, npz_path):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            npz = np.load(npz_path)
            if 'embeds' not in npz:
                return False
            if 'masks' not in npz:
                return False
        except Exception as e:
            logger.error(f'Error loading file: {npz_path}')
            raise e 
        return True 
    
    def load_outputs_npz(self, npz_path):
        data = np.load(npz_path)
        embeds = data['embeds']
        masks = data['masks']
        return [embeds, masks]
    
    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, 
        models: List[Any], 
        text_encoding_strategy: TextEncodingStrategy, 
        infos: List
    ):
        if not self.warn_fp8_weights:
            if models[0].model.lm_head.weight.dtype == torch.float8_e4m3fn:
                logger.warning(
                    "Qwen2VL model is using fp8 weights for caching. This may affect the quality of the cached outputs."
                )
            self.warn_fp8_weights = True
        step1x_edit_encoding_strategy: Step1XEditEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]
        images = [PIL.Image.open(info.ref_absolute_path).convert('RGB') for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions, images)
        with torch.no_grad():
            embs, masks = step1x_edit_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens_and_masks
            )
        if embs.dtype == torch.bfloat16:
            embs = embs.float()
        if masks.dtype == torch.bfloat16:
            masks = masks.float()
        
        for i, info in enumerate(infos):
            emb_i = embs[i]
            mask_i = masks[i]
            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz, 
                    embeds=emb_i.cpu().numpy(), 
                    masks=mask_i.cpu().numpy()
                )
            else:
                info.text_encoder_outputs = (emb_i, mask_i)



from library.train_util import load_image, trim_and_resize_if_required, ImageInfo, IMAGE_TRANSFORMS
# for new_cache_latents
def step1x_load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []

    ref_images: List[torch.Tensor] = []
    ref_alpha_masks: List[np.ndarray] = []

    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        ref_image = load_image(info.ref_absolute_path, use_alpha_mask) if info.ref_image is None else np.array(info.ref_image, np.uint8)
        # thanks for the authors not introducing randomness into cropping.
        image, original_size, crop_ltrb = trim_and_resize_if_required(random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation)
        ref_image, ref_original_size, ref_crop_ltrb = trim_and_resize_if_required(random_crop, ref_image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation)

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
            
            if ref_image.shape[2] == 4:
                ref_alpha_mask = ref_image[:, :, 3]  # [H,W]
                ref_alpha_mask = ref_alpha_mask.astype(np.float32) / 255.0
                ref_alpha_mask = torch.FloatTensor(ref_alpha_mask)  # [H,W]
            else:
                ref_alpha_mask = torch.ones_like(ref_image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
            ref_alpha_mask = None
        alpha_masks.append(alpha_mask)
        ref_alpha_masks.append(ref_alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

        ref_image = ref_image[:, :, :3]
        ref_image = IMAGE_TRANSFORMS(ref_image)
        ref_images.append(ref_image)

    img_tensor = torch.stack(images, dim=0)
    ref_img_tensor = torch.stack(ref_images, dim=0)
    return img_tensor, alpha_masks, ref_img_tensor, ref_alpha_masks, original_sizes, crop_ltrbs

class Step1XEditLatentsCachingStrategy(LatentsCachingStrategy):
    Step1XEdit_LATENTS_NPZ_SUFFIX = "_step1x_latents.npz"
    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
    
    @property
    def cache_suffix(self) -> str:
        return Step1XEditLatentsCachingStrategy.Step1XEdit_LATENTS_NPZ_SUFFIX
    
    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + Step1XEditLatentsCachingStrategy.Step1XEdit_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)  # support multi-resolution
    
    def _default_load_latents_from_disk(
        self, latents_stride: Optional[int], npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        if latents_stride is None:
            key_reso_suffix = ""
        else:
            latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"  # e.g. "_32x64", HxW

        npz = np.load(npz_path)
        if "latents" + key_reso_suffix not in npz:
            raise ValueError(f"latents{key_reso_suffix} not found in {npz_path}")

        latents = npz["latents" + key_reso_suffix]
        ref_latents = npz["ref_latents" + key_reso_suffix]
        original_size = npz["original_size" + key_reso_suffix].tolist()
        crop_ltrb = npz["crop_ltrb" + key_reso_suffix].tolist()
        flipped_latents = npz["latents_flipped" + key_reso_suffix] if "latents_flipped" + key_reso_suffix in npz else None
        ref_flipped_latents = npz["ref_latents_flipped" + key_reso_suffix] if "ref_latents_flipped" + key_reso_suffix in npz else None
        alpha_mask = npz["alpha_mask" + key_reso_suffix] if "alpha_mask" + key_reso_suffix in npz else None
        ref_alpha_mask = npz["ref_alpha_mask" + key_reso_suffix] if "ref_alpha_mask" + key_reso_suffix in npz else None
        return latents, ref_latents, original_size, crop_ltrb, flipped_latents, ref_flipped_latents, alpha_mask, ref_alpha_mask

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._custom_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)

    def _custom_cache_batch_latents(
        self,
        encode_by_vae,
        vae_device,
        vae_dtype,
        image_infos: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool = False,
    ):
        """
        Default implementation for cache_batch_latents. Image loading, VAE, flipping, alpha mask handling are common.
        """
        from library import train_util  # import here to avoid circular import

        img_tensor, alpha_masks, ref_img_tensor, ref_alpha_masks, original_sizes, crop_ltrbs = step1x_load_images_and_masks_for_caching(
            image_infos, alpha_mask, random_crop
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)
        ref_img_tensor = ref_img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
            ref_latents_tensors = encode_by_vae(ref_img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            ref_img_tensor = torch.flip(ref_img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
                ref_flipped_latents = encode_by_vae(ref_img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)
            ref_flipped_latents = [None] * len(ref_latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            ref_latents = ref_latents_tensors[i]
            flipped_latent = flipped_latents[i]
            ref_flipped_latent = ref_flipped_latents[i]
            alpha_mask = alpha_masks[i]
            ref_alpha_mask = ref_alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            latents_size = latents.shape[1:3]  # H, W
            
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}" if multi_resolution else ""  # e.g. "_32x64", HxW

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz, latents, ref_latents, 
                    original_size, crop_ltrb, 
                    flipped_latent, ref_flipped_latent, 
                    alpha_mask, ref_alpha_mask, 
                    key_reso_suffix
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                info.ref_latents = ref_latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                    info.ref_latents_flipped = ref_flipped_latent
                info.alpha_mask = alpha_mask
    
    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        ref_latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        ref_flipped_latents_tensor=None,
        alpha_mask=None,
        ref_alpha_mask=None,
        key_reso_suffix="",
    ):
        kwargs = {}

        if os.path.exists(npz_path):
            # load existing npz and update it
            npz = np.load(npz_path)
            for key in npz.files:
                kwargs[key] = npz[key]

        kwargs["latents" + key_reso_suffix] = latents_tensor.float().cpu().numpy()
        kwargs["ref_latents" + key_reso_suffix] = ref_latents_tensor.float().cpu().numpy()
        kwargs["original_size" + key_reso_suffix] = np.array(original_size)
        kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = flipped_latents_tensor.float().cpu().numpy()
            kwargs["ref_latents_flipped" + key_reso_suffix] = ref_flipped_latents_tensor.float().cpy().numpy()
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = alpha_mask.float().cpu().numpy()
            kwargs["ref_alpha_mask" + key_reso_suffix] = ref_alpha_mask.float().cpu().numpy()
        np.savez(npz_path, **kwargs)