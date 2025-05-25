<div align="center">
  <img src="assets/logo.png"  height=100>
</div>
<div align="center">
  <a href="https://step1x-edit.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2504.17761"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit"><img src="https://img.shields.io/static/v1?label=Model&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://discord.gg/j3qzuAyn"><img src="https://img.shields.io/static/v1?label=Discord%20Channel&message=Discord&color=purple"></a> &ensp;
  
  <a href="https://huggingface.co/stepfun-ai/Step1X-Edit"><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step1X-Edit"><img src="https://img.shields.io/static/v1?label=Online%20Demo&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/datasets/stepfun-ai/GEdit-Bench"><img src="https://img.shields.io/static/v1?label=GEdit-Bench&message=HuggingFace&color=yellow"></a> &ensp;
  [![Run on Replicate](https://replicate.com/zsxkib/step1x-edit/badge)](https://replicate.com/zsxkib/step1x-edit) &ensp;
</div>


## 🔥🔥🔥 News!!
* May 22, 2025: 👋 Step1X-Edit now supports Lora finetuning on a single 24GB GPU now! A hand-fixing Lora for anime characters has also been released. [Download Lora](https://huggingface.co/stepfun-ai/Step1X-Edit)
* Apr 30, 2025: 🎉 Step1X-Edit ComfyUI Plugin is available now, thanks for the community contribution! [quank123wip/ComfyUI-Step1X-Edit](https://github.com/quank123wip/ComfyUI-Step1X-Edit) & [raykindle/ComfyUI_Step1X-Edit](https://github.com/raykindle/ComfyUI_Step1X-Edit).
* Apr 27, 2025: 🎉 With community support, we update the inference code and model weights of Step1X-Edit-FP8. [meimeilook/Step1X-Edit-FP8](https://huggingface.co/meimeilook/Step1X-Edit-FP8) & [rkfg/Step1X-Edit-FP8](https://huggingface.co/rkfg/Step1X-Edit-FP8).
* Apr 26, 2025: 🎉 Step1X-Edit is now live — you can try editing images directly in the online demo! [Online Demo](https://huggingface.co/spaces/stepfun-ai/Step1X-Edit)
* Apr 25, 2025: 👋 We release the evaluation code and benchmark data of Step1X-Edit. [Download GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench)
* Apr 25, 2025: 👋 We release the inference code and model weights of Step1X-Edit. [ModelScope](https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit) & [HuggingFace](https://huggingface.co/stepfun-ai/Step1X-Edit) models.
* Apr 25, 2025: 👋 We have made our technical report available as open source. [Read](https://arxiv.org/abs/2504.17761)

<!-- ## Image Edit Demos -->

<div align="center">
<img width="720" alt="demo" src="assets/image_edit_demo.gif">
<p><b>Step1X-Edit:</b> a unified image editing model performs impressively on various genuine user instructions. </p>
</div>


## 🧩 Community Contributions

If you develop/use Step1X-Edit in your projects, welcome to let us know 🎉.

- FP8 model weights: [meimeilook/Step1X-Edit-FP8](https://huggingface.co/meimeilook/Step1X-Edit-FP8) by [meimeilook](https://huggingface.co/meimeilook);  [rkfg/Step1X-Edit-FP8](https://huggingface.co/rkfg/Step1X-Edit-FP8) by [rkfg](https://huggingface.co/rkfg)
- Step1X-Edit ComfyUI Plugin: [quank123wip/ComfyUI-Step1X-Edit](https://github.com/quank123wip/ComfyUI-Step1X-Edit) by [quank123wip](https://github.com/quank123wip); [raykindle/ComfyUI_Step1X-Edit](https://github.com/raykindle/ComfyUI_Step1X-Edit) by [raykindle](https://github.com/raykindle)
- Training scripts: [hobart07/Step1X-Edit_train](https://github.com/hobart07/Step1X-Edit_train) by [hobart07](https://github.com/hobart07)

## 📑 Open-source Plan
- [x] Inference & Checkpoints
- [x] Online demo (Gradio)
- [x] Fine-tuning scripts
- [ ] Diffusers 
- [ ] Multi-gpus Sequence Parallel inference
- [x] FP8 Quantified weight
- [x] ComfyUI



## 1. Introduction
we introduce a state-of-the-art image editing model, **Step1X-Edit**, which aims to provide comparable performance against the closed-source models like GPT-4o and Gemini2 Flash. 
More specifically, we adopt the Multimodal LLM to process the reference image and user's editing instruction. A latent embedding has been extracted and integrated with a diffusion image decoder to obtain  the target image. To train the model, we build a data generation pipeline to produce a high-quality dataset. 
For evaluation, we develop the GEdit-Bench, a novel benchmark rooted in real-world user instructions. Experimental results on GEdit-Bench demonstrate that Step1X-Edit outperforms existing open-source baselines by a substantial margin and approaches the performance of leading proprietary models, thereby making significant contributions to the field of image editing. 
More details please refer to our [technical report](https://arxiv.org/abs/2504.17761).


## 2. Model Usage
### 2.1  Requirements

The following table shows the requirements for running Step1X-Edit model (batch size = 1, w/o cfg distillation) to edit images:

|     Model    |     Peak GPU Memory (512 / 786 / 1024)  | 28 steps w flash-attn(512 / 786 / 1024) |
|:------------:|:------------:|:------------:|
| Step1X-Edit   |                42.5GB / 46.5GB / 49.8GB  | 5s / 11s / 22s |
| Step1X-Edit-FP8   |             31GB / 31.5GB / 34GB     | 6.8s / 13.5s / 25s | 
| Step1X-Edit + offload   |       25.9GB / 27.3GB / 29.1GB | 49.6s / 54.1s / 63.2s |
| Step1X-Edit-FP8 + offload   |   18GB / 18GB / 18GB | 35s / 40s / 51s |

* The model is tested on one H800 GPUs.
* We recommend to use GPUs with 80GB of memory for better generation quality and efficiency.


### 2.2 Dependencies and Installation


python >=3.10.0 and install [torch](https://pytorch.org/get-started/locally/) >= 2.2 with cuda toolkit and corresponding torchvision. We test our model using torch==2.3.1 and torch==2.5.1 with cuda-12.1.


Install requirements:
  
``` bash
pip install -r requirements.txt
```

Install [`flash-attn`](https://github.com/Dao-AILab/flash-attention), here we provide a script to help find the pre-built wheel suitable for your system. 
    
```bash
python scripts/get_flash_attn.py
```

The script will generate a wheel name like `flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`, which could be found in [the release page of flash-attn](https://github.com/Dao-AILab/flash-attention/releases).

Then you can download the corresponding pre-built wheel and install it following the instructions in [`flash-attn`](https://github.com/Dao-AILab/flash-attention).


### 2.3 Inference Scripts
After downloading the [model weights](https://huggingface.co/stepfun-ai/Step1X-Edit), you can use the following scripts to edit images:

```
bash scripts/run_examples.sh
```
The default script runs the inference code with non-quantified weights. If you want to save the GPU memory usage, you can 1) download the FP8 weights and set the `--quantized` flag in the script, or 2) set the `--offload` flag in the script to offload some modules to CPU.

This default script runs the inference code on example inputs. The results will look like:
<div align="center">
<img width="1080" alt="results" src="assets/results_show.png">
</div>

### 2.4 Gradio Scripts

Change the `model_path` in `gradio_app.py` to the local path of Step1X-Edit. Then run

```bash
python gradio_app.py
```

Then the gradio demo will run on `localhost:32800`.

## 3. Finetuning

### 3.1 Training scripts

The script `./scripts/finetuning.sh` shows how to fine-tune the Step1X-Edit model. With our default strategy, it is possible to fine-tune Step1X-Edit with 1024 resolution on a single 24GB GPU. Our fine-tuning script is adapted from  [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).

```bash
bash ./scripts/finetuning.sh
```

The custom dataset is organized by `./library/data_configs/step1x_edit.toml`. Here `metadata_file` contains all the training sampels, including the absolute paths of source images, absolute paths of target images and instructions.

The `metadata_file` should be a json file containing a dict as follows:

```
{
  <target image path, str>: {
    'ref_image_path': <source image path, str>
    'caption': <the editing instruction, str>
  }, 
  ...
}
```

### 3.2 Inference with Lora

Simply add `--lora <path to your lora weights>` when using `inference.py`. For example:

```bash
python inference.py --input_dir ./examples \
    --model_path /data/work_dir/step1x-edit/ \
    --json_path ./examples/prompt_cn.json \
    --output_dir ./output_cn \
    --seed 1234 --size_level 1024 \
    --lora 20250521_001-lora256-alpha128-fix-hand-per-epoch/step1x-edit_test.safetensors
```

To reproduce the cases below, 

```bash 
bash scripts/run_examples_fix_hand.sh
```


### 3.3 Performances

Here is the the GPU memory cost during training with lora rank as 64 and batchsize as 1:

|     Precision of DiT    |     bf16 (512 / 786 / 1024)  | fp8 (512 / 786 / 1024) |
|:------------:|:------------:|:------------:|
| GPU Memory   |                29.7GB / 31.6GB / 33.8GB  | 19.8GB / 21.3GB / 23.6GB |

Here is an example for our [pretrained Lora weights](https://huggingface.co/stepfun-ai/Step1X-Edit/tree/main/lora), which is designed for fixing corrupted hands of anime characters.

<div align="center">
<img width="1080" alt="results" src="assets/lora_teaser.png">
</div>

## 4. Benchmark
We release [GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) as a new benchmark, grounded in real-world usages is developed to support more authentic and comprehensive evaluation. This benchmark, which is carefully curated to reflect actual user editing needs and a wide range of editing scenarios, enables more authentic and comprehensive evaluations of image editing models.
The evaluation process and related code can be found in [GEdit-Bench/EVAL.md](GEdit-Bench/EVAL.md). Part results of the benchmark are shown below:
<div align="center">
<img width="1080" alt="results" src="assets/eval_res_en.png">
</div>


## 5. Citation
```
@article{liu2025step1x-edit,
      title={Step1X-Edit: A Practical Framework for General Image Editing}, 
      author={Shiyu Liu and Yucheng Han and Peng Xing and Fukun Yin and Rui Wang and Wei Cheng and Jiaqi Liao and Yingming Wang and Honghao Fu and Chunrui Han and Guopeng Li and Yuang Peng and Quan Sun and Jingwei Wu and Yan Cai and Zheng Ge and Ranchen Ming and Lei Xia and Xianfang Zeng and Yibo Zhu and Binxing Jiao and Xiangyu Zhang and Gang Yu and Daxin Jiang},
      journal={arXiv preprint arXiv:2504.17761},
      year={2025}
}
```

## 6. Acknowledgement
We would like to express our sincere thanks to the contributors of [Kohya](https://github.com/kohya-ss/sd-scripts/tree/sd3), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Qwen](https://github.com/QwenLM/Qwen2.5), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) teams, for their open research and exploration.


## 7. Disclaimer
The results produced by this image editing model are entirely determined by user input and actions. The development team and this open-source project are not responsible for any outcomes or consequences arising from its use.

## 8. LICENSE
Step1X-Edit is licensed under the Apache License 2.0. You can find the license files in the respective github and  HuggingFace repositories.
