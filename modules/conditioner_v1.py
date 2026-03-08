import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

Qwen25VL_7b_PREFIX = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''


def split_string(s):
    # 将中文引号替换为英文引号
    s = s.replace("“", '"').replace("”", '"')  # use english quotes
    result = []
    # 标记是否在引号内
    in_quotes = False
    temp = ""

    # 遍历字符串中的每个字符及其索引
    for idx, char in enumerate(s):
        # 如果字符是引号且索引大于 155
        if char == '"' and idx > 155:
            # 将引号添加到临时字符串
            temp += char
            # 如果不在引号内
            if not in_quotes:
                # 将临时字符串添加到结果列表
                result.append(temp)
                # 清空临时字符串
                temp = ""

            # 切换引号状态
            in_quotes = not in_quotes
            continue
        # 如果在引号内
        if in_quotes:
            # 如果字符是空格
            if char.isspace():
                pass  # have space token

            # 将字符用中文引号包裹后添加到结果列表
            result.append("“" + char + "”")
        else:
            # 将字符添加到临时字符串
            temp += char

    # 如果临时字符串不为空
    if temp:
        # 将临时字符串添加到结果列表
        result.append(temp)

    return result

class Qwen25VL_7b_Embedder(torch.nn.Module):
    def __init__(self, model_path, max_length=640, dtype=torch.bfloat16, device='cuda'):
        super(Qwen25VL_7b_Embedder, self).__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",#目前还没装好
        ).to(torch.cuda.current_device())

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path,min_pixels = 256 * 28 * 28, max_pixels = 324 * 28 * 28)

        self.prefix = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''

    def forward(self, caption, ref_images):
        text_list=caption
        embs = torch.zeros(len(text_list),self.max_length, self.model.config.hidden_size, dtype=torch.bfloat16, device=torch.cuda.current_device())
        hidden_states = torch.zeros(len(text_list),self.max_length, self.model.config.hidden_size, dtype=torch.bfloat16, device=torch.cuda.current_device())
        masks = torch.zeros(len(text_list),self.max_length, dtype=torch.long, device=torch.cuda.current_device())
        input_ids_list = []
        attention_mask_list = []
        emb_list = []

        def split_string(s):
            s = s.replace("'", '"').replace("“", '"').replace("”", '"')  # use english quotes
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

        for idx, (txt, imgs) in enumerate(zip(text_list, ref_images)):
            # image_list = []
            # for idx_img in imgs:
            #     image_list.append(to_pil(idx_img))
            
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]

            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})
            # 先添加所有的 image
            # messages[0]["content"].extend([{"type": "image", "image": img} for img in image_list])
            messages[0]['content'].append({"type": "image", "image": to_pil(imgs)})

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
                token_each=txt_inputs.input_ids
                if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                    token_each = token_each[:,1:-1]
                    token_list.append(token_each)
                else:
                    token_list.append(token_each)

            new_txt_ids=torch.cat(token_list,dim=1).to("cuda")

            new_txt_ids = new_txt_ids.to(old_inputs_ids.device)
            idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
            idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
            inputs.input_ids = torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]],dim=0).unsqueeze(0).to("cuda")
            inputs.attention_mask= (inputs.input_ids>0).long().to("cuda")
            outputs = self.model(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, pixel_values = inputs.pixel_values.to("cuda"), image_grid_thw = inputs.image_grid_thw.to("cuda"), output_hidden_states=True)
            # outputs = self.model.base_model(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, pixel_values = inputs.pixel_values.to("cuda"), image_grid_thw = inputs.image_grid_thw.to("cuda"), output_hidden_states=True)

            emb = outputs['hidden_states'][-1]
            # hidden_state = output['hidden_states'][8]

            embs[idx,:min(self.max_length,emb.shape[1]-217)] = emb[0,217:][:self.max_length]
            # hidden_states[idx,:min(self.max_length,hidden_state.shape[1]-217)] = hidden_state[0,217:][:self.max_length]

            masks[idx,:min(self.max_length,emb.shape[1]-217)]=torch.ones((min(self.max_length,emb.shape[1]-217)), dtype=torch.long, device=torch.cuda.current_device())

        return embs,masks



class Qwen25VL_7b_Embedder_backup(torch.nn.Module):
    def __init__(self, model_path, max_length=640, dtype=torch.bfloat16, device="cuda"):
        super(Qwen25VL_7b_Embedder_backup, self).__init__()
        self.max_length = max_length

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        ).to(torch.cuda.current_device())

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28
        )

        self.prefix = Qwen25VL_7b_PREFIX

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, caption, ref_images):
        text_list = caption
        embs = torch.zeros(
            len(text_list),
            self.max_length,
            self.model.config.hidden_size,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        hidden_states = torch.zeros(
            len(text_list),
            self.max_length,
            self.model.config.hidden_size,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        masks = torch.zeros(
            len(text_list),
            self.max_length,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        input_ids_list = []
        attention_mask_list = []
        emb_list = []

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

        for idx, (txt, imgs) in enumerate(zip(text_list, ref_images)):

            messages = [{"role": "user", "content": []}]

            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})

            messages[0]["content"].append({"type": "image", "image": to_pil(imgs)})

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
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values.to("cuda"),
                image_grid_thw=inputs.image_grid_thw.to("cuda"),
                output_hidden_states=True,
            )

            emb = outputs["hidden_states"][-1]

            embs[idx, : min(self.max_length, emb.shape[1] - 217)] = emb[0, 217:][
                : self.max_length
            ]

            masks[idx, : min(self.max_length, emb.shape[1] - 217)] = torch.ones(
                (min(self.max_length, emb.shape[1] - 217)),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )

        return embs, masks