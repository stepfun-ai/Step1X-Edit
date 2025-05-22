python inference.py --input_dir ./examples \
    --model_path /data/work_dir/step1x-edit/ \
    --json_path ./examples/prompt_fix_hand.json \
    --output_dir ./output_fix_hand \
    --seed 1234 --size_level 1024 --lora step1x-edit-lora256-alpha128-fix-hand.safetensors