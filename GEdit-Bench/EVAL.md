# EVAL

Our evaluation process consists of the following steps:
1. Prepare the Environment and Dataset
   - Install required dependencies:
     ```bash
     conda env create -f qwen25vl_environment.yml
     conda activate qwen25vl
     ```
   - Set up your API keys in secret_t2.env for GPT4.1 access
   - Then download our dataset stepfun-ai/GEdit-Bench:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("stepfun-ai/GEdit-Bench")
     ```

2. Generate and Organize Your Images
   - Generate images following the example code in `generate_image_example.py`
   - Organize your generated images in the following directory structure:
     ```
     results/
     ├── method_name/
     │   └── fullset/
     │       └── edit_task/
     │           ├── cn/  # Chinese instructions
     │           │   ├── key1.png
     │           │   ├── key2.png
     │           │   └── ...
     │           └── en/  # English instructions
     │               ├── key1.png
     │               ├── key2.png
     │               └── ...
     ```

3. Evaluate using GPT4.1/Qwen2.5VL-72B-Instruct-AWQ
   - For GPT-4.1 evaluation:
     ```bash
     python test_gedit_score.py --model_name your_method --save_path --backbone gpt4o
     ```
   - For Qwen evaluation:
     ```bash
     python test_gedit_score.py --model_name your_method --save_path --backbone qwen25vl
     ```

4. Analyze your results and obtain scores across all dimensions
   - Run the analysis script to get scores for semantics, quality, and overall performance:
     ```bash
     python calculate_statistics.py --model_name your_method --save_path /path/to/results --backbone gpt4o
     ```
   - This will output scores broken down by edit category and provide aggregate metrics

# Acknowledgements

This project builds upon and adapts code from the following excellent repositories:

- [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore): A visual instruction-guided explainable metric for evaluating conditional image synthesis

We thank the authors of these repositories for making their code publicly available.

