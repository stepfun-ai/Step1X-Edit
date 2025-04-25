# Standalone VIEScore

A Standalone version for VIEScore that can be further developed easily.

## Project Structure
* `mllm_tools`: Plug-and-Play MLLMs codebase.
* `prompts_raw`: Storing the prompt template.
* `parse_prompt.py` Parsing prompt templates in the `prompts_raw` folder
* `vie_prompts.py`: generated automatically through `parse_prompt.py`
* `__init__.py` Implementation of VIEScore.

## Running VIEScore
![](https://tiger-ai-lab.github.io/VIEScore/static/images/method.png)
```python
from viescore import VIEScore
backbone = "gemini"
vie_score = VIEScore(backbone=backbone, task="t2v")

score_list = vie_score.evaluate(pil_image, text_prompt)
sementics_score, quality_score, overall_score = score_list
```

Currently only support T2I, Image Editing, and T2V tasks. You can refer to the `paper_implementation` for other tasks in the paper.


## Citation

Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@misc{ku2023viescore,
                title={VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation}, 
                author={Max Ku and Dongfu Jiang and Cong Wei and Xiang Yue and Wenhu Chen},
                year={2023},
                eprint={2312.14867},
                archivePrefix={arXiv},
                primaryClass={cs.CV}
            }
```
