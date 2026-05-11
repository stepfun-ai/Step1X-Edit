set -euo pipefail

echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -n1 | cut -d',' -f1 | tr -d '
')
echo "Auto-selected GPU_ID=${GPU_ID}"

GPU_ID="${GPU_ID:-1}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" accelerate launch \
    --main_process_port 29502 \
    --mixed_precision bf16 \
    --num_cpu_threads_per_process 1 \
    --num_processes 1 \
    --config_file ./library/accelerate_config.yaml \
    finetuning.py \
    --pretrained_model_name_or_path /scratch3/f007yzf/models/step1x_v11/step1x-edit-i1258.safetensors \
    --qwen2p5vl /scratch3/f007yzf/models/step1x_v11/Qwen2.5-VL-7B-Instruct \
    --ae /scratch3/f007yzf/models/step1x_v11/vae.safetensors \
    --cache_latents_to_disk \
    --save_model_as safetensors \
    --sdpa \
    --persistent_data_loader_workers \
    --max_data_loader_n_workers 2 \
    --seed 20260307 \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --save_precision bf16 \
    --network_module library.qwen_connector_module_v1 \
    --network_train_unet_only \
    --optimizer_type adamw8bit \
    --learning_rate 1e-4 \
    --cache_text_encoder_outputs \
    --cache_text_encoder_outputs_to_disk \
    --highvram \
    --max_train_epochs 1000 \
    --save_every_n_epochs 100 \
    --dataset_config library/data_configs/step1x_edit.toml \
    --output_dir /scratch3/f007yzf/repos/Step1X-Edit-clean/output \
    --output_name step1x-edit-qwen-connector-v1 \
    --timestep_sampling shift \
    --discrete_flow_shift 3.1582 \
    --model_prediction_type raw \
    --guidance_scale 1.0 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 
