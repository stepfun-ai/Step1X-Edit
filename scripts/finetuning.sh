accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 --num_processes 1 \
--config_file ./library/accelerate_config.yaml \
finetuning.py \
--pretrained_model_name_or_path <path to step1x-edit-i1258.safetensors> \
--qwen2p5vl <path to your qwen2.5vl> \
--ae <path to vae.safetensors> \
--cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
--max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
--network_module library.lora_module --network_dim 64 --network_alpha 32 --network_train_unet_only \
--optimizer_type adamw8bit --learning_rate 1e-4 \
--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
--highvram --max_train_epochs 100 --save_every_n_epochs 5 --dataset_config library/data_configs/step1x_edit.toml \
--output_dir <output directory> \
--output_name step1x-edit_test \
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --fp8_base