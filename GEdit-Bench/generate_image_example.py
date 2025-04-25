from datasets import Dataset, load_dataset
import math

# Dataset info structure:
# - task_type: string - Type of the task
# - key: string - Unique identifier for the sample
# - instruction: string - Task instruction/prompt
# - instruction_language: string - Language of the instruction
# - input_image: Image - Original input image
# - input_image_raw: Image - Raw/unprocessed input image
# - Intersection_exist: bool - Whether intersection exists

def calculate_dimensions(target_area, ratio):
    # 根据长宽比计算宽度和高度
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    
    # 确保宽度和高度都能被16整除
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    
    # 重新计算面积以确保尽可能接近目标面积
    new_area = width * height
    if new_area < target_area:
        width += 32
        new_area = width * height
    elif new_area > target_area:
        width -= 32
        new_area = width * height
    
    return width, height, new_area

dataset = load_dataset("stepfun-ai/GEdit-Bench")
save_path = "your_save_dir/modelname/"

for item in dataset:
    task_type = item['task_type']
    key = item['key']
    instruction = item['instruction']
    instruction_language = item['instruction_language']
    input_image = item['input_image']
    input_image_raw = item['input_image_raw']
    intersection_exist = item['Intersection_exist']

    target_width, target_height, new_are = calculate_dimensions(512 * 512, input_image_raw.width / input_image_raw.height)
    resize_input_image = input_image_raw.resize((target_width, target_height))

    save_path_fullset_source_image = f"{save_path}/fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png"
    save_path_fullset = f"{save_path}/fullset/{task_type}/{instruction_language}/{key}.png"
    
    input_image.save(save_path_fullset_source_image)
    resize_input_image.save(save_path_fullset)