import argparse
import datetime
import json 
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms import functional as F
from tqdm import tqdm 

import sampling
from modules.autoencoder import AutoEncoder
from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from modules.model_edit import Step1XParams, Step1XEdit

def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model


def load_models(
    dit_path=None,
    ae_path=None,
    qwen2vl_model_path=None,
    device="cuda",
    max_length=256,
    dtype=torch.bfloat16,
):
    qwen2vl_encoder = Qwen2VLEmbedder(
        qwen2vl_model_path,
        device=device,
        max_length=max_length,
        dtype=dtype,
    )

    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        step1x_params = Step1XParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
        )
        dit = Step1XEdit(step1x_params)

    ae = load_state_dict(ae, ae_path, 'cpu')
    dit = load_state_dict(
        dit, dit_path, 'cpu'
    )

    ae = ae.to(dtype=torch.float32)

    return ae, dit, qwen2vl_encoder

def equip_dit_with_lora_sd_scripts(ae, text_encoders, dit, lora, device='cuda'):
    from safetensors.torch import load_file
    weights_sd = load_file(lora)
    is_lora = True
    from library import lora_module
    module = lora_module
    lora_model, _ = module.create_network_from_weights(1.0, None, ae, text_encoders, dit, weights_sd, True)
    lora_model.merge_to(text_encoders, dit, weights_sd)

    lora_model.set_multiplier(1.0)
    return lora_model

class ImageGenerator:
    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwen2vl_model_path=None,
        device="cuda",
        max_length=640,
        dtype=torch.bfloat16,
        quantized=False,
        offload=False,
        lora=None,
    ) -> None:
        self.device = torch.device(device)
        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            dtype=dtype,
            device=self.device
        )
        if not quantized:
            self.dit = self.dit.to(dtype=torch.bfloat16)
        else:
            self.dit = self.dit.to(dtype=torch.float8_e4m3fn)
        if not offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)
        self.quantized = quantized 
        self.offload = offload
        if lora is not None:
            self.lora_module = equip_dit_with_lora_sd_scripts(
                self.ae,
                [self.llm_encoder],
                self.dit,
                lora,
                device=self.dit.device,
            )
        else:
            self.lora_module = None


    def prepare(self, prompt, img, ref_image, ref_image_raw):
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)

        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)

        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]
        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
        txt, mask = self.llm_encoder(prompt, ref_image_raw)
        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)


        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float = 4.5,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
    ):
        if self.offload:
            self.dit = self.dit.to(self.device)
        if show_progress:
            pbar = tqdm(itertools.pairwise(timesteps), desc='denoising...')
        else:
            pbar = itertools.pairwise(timesteps)
        for t_curr, t_prev in pbar:
            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )
            # txt, vec = self.dit.connector(llm_embedding, t_vec, mask)


            # pred = self.dit(
            #     img=img,
            #     img_ids=img_ids,
            #     txt=txt,
            #     txt_ids=txt_ids,
            #     y=vec,
            #     timesteps=t_vec,
            # )

            if cfg_guidance != -1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                if t_curr > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                    pred = uncond + cfg_guidance * (
                        cond - uncond
                    ) / self.process_diff_norm(diff_norm, k=0.4)
                else:
                    pred = uncond + cfg_guidance * (cond - uncond)
            tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred
            img_input_length = img.shape[1] // 2
            img = torch.cat(
                [
                tem_img[:, :img_input_length],
                img[ : img.shape[0] // 2, img_input_length:],
                ], dim=1
            )
        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        return img[:, :img.shape[1] // 2]

    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def output_process_image(self, resize_img, image_size):
        res_image = resize_img.resize(image_size)
        return res_image
    
    def input_process_image(self, img, img_size=512):
        # 1. 打开图片
        w, h = img.size
        r = w / h 

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size

    @torch.inference_mode()
    def generate_image(
        self,
        prompt,
        negative_prompt,
        ref_images,
        num_steps,
        cfg_guidance,
        seed,
        num_samples=1,
        init_image=None,
        image2image_strength=0.0,
        show_progress=False,
        size_level=512,
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        
        width, height = ref_images_raw.width, ref_images_raw.height


        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)
        if self.offload:
            self.ae = self.ae.to(self.device)
        ref_images = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        t0 = time.perf_counter()

        if init_image is not None:
            init_image = self.load_image(init_image)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae = self.ae.to(self.device)
            init_image = self.ae.encode(init_image.to() * 2 - 1)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()
        
        x = torch.randn(
            num_samples,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        timesteps = sampling.get_schedule(
            num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
        )

        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        x = torch.cat([x, x], dim=0)
        ref_images = torch.cat([ref_images, ref_images], dim=0)
        ref_images_raw = torch.cat([ref_images_raw, ref_images_raw], dim=0)
        inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.denoise(
                **inputs,
                cfg_guidance=cfg_guidance,
                timesteps=timesteps,
                show_progress=show_progress,
                timesteps_truncate=1.0,
            )
        x = self.unpack(x.float(), height, width)
        if self.offload:
            self.ae = self.ae.to(self.device)
        x = self.ae.decode(x)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()
        x = x.clamp(-1, 1)
        x = x.mul(0.5).add(0.5)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")
        images_list = []
        for img in x.float():
            images_list.append(self.output_process_image(F.to_pil_image(img), img_info))
        return images_list


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output image directory')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing image names and prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of diffusion steps')
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG guidance strength')
    parser.add_argument('--size_level', default=512, type=int)
    parser.add_argument('--offload', action='store_true', help='Use offload for large models')
    parser.add_argument('--quantized', action='store_true', help='Use fp8 model weights')
    parser.add_argument('--lora', type=str, default=None)
    args = parser.parse_args()

    assert os.path.exists(args.input_dir), f"Input directory {args.input_dir} does not exist."
    assert os.path.exists(args.json_path), f"JSON file {args.json_path} does not exist."

    args.output_dir = args.output_dir.rstrip('/') + ('-offload' if args.offload else "") + ('-quantized' if args.quantized else "") + f"-{args.size_level}"
    os.makedirs(args.output_dir, exist_ok=True)

    image_and_prompts = json.load(open(args.json_path, 'r'))

    image_edit = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=os.path.join(args.model_path, 'Qwen2.5-VL-7B-Instruct'),
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        lora=args.lora,
    )

    time_list = []
    for image_name, prompt in image_and_prompts.items():
        image_path = os.path.join(args.input_dir, image_name)
        output_path = os.path.join(args.output_dir, image_name)
        start_time = time.time()

        image = image_edit.generate_image(
            prompt,
            negative_prompt="",
            ref_images=Image.open(image_path).convert("RGB"),
            num_samples=1,
            num_steps=args.num_steps,
            cfg_guidance=args.cfg_guidance,
            seed=args.seed,
            show_progress=True,
            size_level=args.size_level,
        )[0]
        
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        time_list.append(time.time() - start_time)

        image.save(
            os.path.join(output_path), lossless=True
        )
    print(f'average time for {args.output_dir}: ', sum(time_list[1:]) / len(time_list[1:]))


if __name__ == "__main__":
    main()
