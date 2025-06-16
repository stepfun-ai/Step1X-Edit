import json
import multiprocessing
from inference import *
from tqdm import tqdm
import torch
from multiprocessing import Process, set_start_method

def process_subset(process_id, subset_data):

    device = 'cuda:'+str(process_id) if torch.cuda.is_available() else 'cpu'
    print(device)
  
    stepedit = ImageGenerator(device=device,lora ='path_to_your_lora')

    # 使用tqdm处理当前子列表
    for dic in tqdm(subset_data, desc=f"Process {process_id} on GPU {process_id}"):
        input_folder = dic['input_folder']
        output_folder = dic['output_folder']
        # try:
        input
        stepedit.inference(input_path = 'path_to_your_inputimage', output_path = 'path_to_your_outputimg',prompt='your_prompt')


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    # 读取数据
    '''
     the json file shuld like this :
     {
     {    "input_file":"",
          "output_file":""  
        }
     }
    '''
    with open('your_inference_json', 'r') as f:
        data = json.load(f)

    # 进程数量
    num_processes = 8

    # 计算每个进程处理的数据量
    chunk_size = len(data) // num_processes
    # 处理余数
    remainder = len(data) % num_processes

    # 将数据分割成8份
    subsets = []
    start_idx = 0
    for i in range(num_processes):
        # 最后一个进程处理剩余的所有数据
        if i == num_processes - 1:
            subsets.append(data[start_idx:])
        else:
            # 前面的进程平均分配数据，考虑余数
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            subsets.append(data[start_idx:end_idx])
            start_idx = end_idx

    # 创建并启动进程
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=process_subset, args=(i, subsets[i]))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All processes completed.")
