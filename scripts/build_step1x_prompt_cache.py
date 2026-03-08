from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


REPO_ROOT = Path("/scratch3/f007yzf/repos/Step1X-Edit-clean")
DATA_ROOT = REPO_ROOT / "training_data"
EXPERIMENT_ROOT = REPO_ROOT / "training_6k"
MODEL_PATH = Path("/scratch3/f007yzf/models/step1x_v11/Qwen2.5-VL-7B-Instruct")
OUTPUT_METADATA_DIR = EXPERIMENT_ROOT / "metadata"
EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_METADATA_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 150
MAX_LENGTH = 640
MASTER_SEED = 20260307
SET1_SEED = MASTER_SEED
SET2_SEED = MASTER_SEED + 1
PAIRS = [f"{i:02d}" for i in range(5)]
PREFIXES = ["data", "data_seed_2", "data_seed_3", "data_seed_4", "data_seed_5"]

SINGLE_IMG1_PATH = DATA_ROOT / "source_img" / "data__p0000_pair00_s0__source.png"
SINGLE_IMG2_PATH = DATA_ROOT / "reference_img" / "left" / "data__p0000_pair00_s0.png"
SINGLE_PREVIEW_OUTPUT = OUTPUT_METADATA_DIR / "llm_prompt_preview_v1.json"

DUAL_IMAGE_CAPTION_PROMPT = """You are analyzing facial expressions for a controlled editing task.
Given:
- Image 1: source face to be edited
- Image 2: target expression reference

Output a structured expression editing plan
"""

EMBEDDER_PREFIX = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.

Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.

Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:"""

IMAGE_RE = re.compile(r"^(data(?:_seed_\d+)?)__p(\d+)_pair(\d+)_s(\d+)(?:__(source|target))?\.png$")

ROOTS = {
    "flux_neg": DATA_ROOT / "source_img",
    "flux_pos": DATA_ROOT / "target_img",
    "iy_pos": DATA_ROOT / "reference_img" / "left",
    "iy_neg": DATA_ROOT / "reference_img" / "right",
}


print("DEVICE =", DEVICE)
print("MODEL_PATH =", MODEL_PATH)
print("OUTPUT_METADATA_DIR =", OUTPUT_METADATA_DIR)
print("EXPERIMENT_ROOT =", EXPERIMENT_ROOT)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256 * 28 * 28,
    max_pixels=324 * 28 * 28,
)
print("Loading Qwen2.5-VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)
model.requires_grad_(False)
model.eval()
HIDDEN_SIZE = model.config.hidden_size
print("Model loaded. hidden_size =", HIDDEN_SIZE)


def load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def split_string_for_embedder(text: str) -> List[str]:
    text = text.replace("'", '"').replace("“", '"').replace("”", '"')
    result = []
    in_quotes = False
    temp = ""
    for idx, char in enumerate(text):
        if char == '"' and idx > 155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""
            in_quotes = not in_quotes
            continue
        if in_quotes:
            result.append("“" + char + "”")
        else:
            temp += char
    if temp:
        result.append(temp)
    return result


@torch.inference_mode()
def build_dual_image_prompt(
    img1_path: Path, img2_path: Path, user_prompt: str = "", max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    img1 = load_pil(img1_path)
    img2 = load_pil(img2_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": DUAL_IMAGE_CAPTION_PROMPT},
            {"type": "text", "text": "\n[Source Image (Structure/Identity)]:"},
            {"type": "image", "image": img1},
            {"type": "text", "text": "\n[Reference Image (Style/Expression)]:"},
            {"type": "image", "image": img2},
            {"type": "text", "text": f"\nUser prompt: {user_prompt}"},
            {"type": "text", "text": "\nPlease generate the structured editing plan now."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    return generated_text or user_prompt


@torch.inference_mode()
def build_prompt_embedding(img1_path: Path, prompt: str, max_length: int = MAX_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    img1 = load_pil(img1_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": EMBEDDER_PREFIX},
            {"type": "image", "image": img1},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    old_input_ids = inputs.input_ids
    token_list = []
    for text_each in split_string_for_embedder(text):
        txt_inputs = processor(text=text_each, images=None, videos=None, padding=True, return_tensors="pt")
        token_each = txt_inputs.input_ids
        if token_each[0][0] == 2073 and token_each[0][-1] == 854:
            token_each = token_each[:, 1:-1]
        token_list.append(token_each)
    new_txt_ids = torch.cat(token_list, dim=1).to(old_input_ids.device)
    idx1 = (old_input_ids == 151653).nonzero(as_tuple=True)[1][0]
    idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
    input_ids = torch.cat([old_input_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0).unsqueeze(0)
    attention_mask = (input_ids > 0).long()
    outputs = model(
        input_ids=input_ids.to(DEVICE),
        attention_mask=attention_mask.to(DEVICE),
        pixel_values=inputs.pixel_values.to(DEVICE),
        image_grid_thw=inputs.image_grid_thw.to(DEVICE),
        output_hidden_states=True,
    )
    emb = outputs.hidden_states[-1]
    embeds = torch.zeros((max_length, HIDDEN_SIZE), dtype=torch.bfloat16, device=DEVICE)
    masks = torch.zeros((max_length,), dtype=torch.long, device=DEVICE)
    usable = max(0, emb.shape[1] - 217)
    length = min(max_length, usable)
    if length > 0:
        embeds[:length] = emb[0, 217:217 + length]
        masks[:length] = 1
    return embeds.to(torch.float32).cpu().numpy(), masks.cpu().numpy()


@torch.inference_mode()
def run_single_preview(
    img1_path: Path = SINGLE_IMG1_PATH,
    img2_path: Path = SINGLE_IMG2_PATH,
    output_json: Path = SINGLE_PREVIEW_OUTPUT,
):
    prompt = build_dual_image_prompt(img1_path, img2_path)
    embeds, masks = build_prompt_embedding(img1_path, prompt)
    payload = {
        "img1_path": str(img1_path.resolve()),
        "img2_path": str(img2_path.resolve()),
        "prompt": prompt,
        "embedding_shape": list(embeds.shape),
        "mask_sum": int(masks.sum()),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def candidate_filename(prefix: str, person_id: str, pair_id: str, kind: str) -> str:
    stem = f"{prefix}__p{person_id}_pair{pair_id}_s0"
    if kind == "flux_neg":
        return stem + "__source.png"
    if kind == "flux_pos":
        return stem + "__target.png"
    if kind in {"iy_pos", "iy_neg"}:
        return stem + ".png"
    raise KeyError(kind)


def resolve_image_paths(prefix: str, person_id: str, pair_id: str) -> Dict[str, Path]:
    return {kind: ROOTS[kind] / candidate_filename(prefix, person_id, pair_id, kind) for kind in ROOTS}


def mirror_path(original_path: Path) -> Path:
    rel = original_path.relative_to(DATA_ROOT)
    return EXPERIMENT_ROOT / rel


def ensure_mirror_file(original_path: Path) -> Path:
    mirrored_path = mirror_path(original_path)
    mirrored_path.parent.mkdir(parents=True, exist_ok=True)
    if mirrored_path.exists():
        return mirrored_path
    try:
        mirrored_path.symlink_to(original_path)
    except OSError:
        shutil.copy2(original_path, mirrored_path)
    return mirrored_path


def person_complete(prefix: str, person_id: str) -> bool:
    for pair_id in PAIRS:
        paths = resolve_image_paths(prefix, person_id, pair_id)
        if not all(path.exists() for path in paths.values()):
            return False
    return True


def scan_candidates() -> Dict[str, List[str]]:
    seen = defaultdict(set)
    for root in ROOTS.values():
        for path in root.glob("*.png"):
            m = IMAGE_RE.match(path.name)
            if not m:
                continue
            prefix, person_id, pair_id, s, _ = m.groups()
            if prefix not in PREFIXES or s != "0":
                continue
            seen[prefix].add(person_id)
    candidates = {}
    for prefix in PREFIXES:
        valid = sorted(pid for pid in seen[prefix] if person_complete(prefix, pid))
        candidates[prefix] = valid
    return candidates


def build_person_to_prefixes(candidates_by_prefix: Dict[str, List[str]]) -> Dict[str, List[str]]:
    person_to_prefixes = defaultdict(list)
    for prefix, persons in candidates_by_prefix.items():
        for pid in persons:
            person_to_prefixes[pid].append(prefix)
    return {pid: sorted(prefixes) for pid, prefixes in person_to_prefixes.items()}


def hopcroft_karp(graph: Dict[str, List[str]], left_nodes: List[str], right_nodes: List[str]):
    inf = 10**9
    pair_u = {u: None for u in left_nodes}
    pair_v = {v: None for v in right_nodes}
    dist = {}

    from collections import deque

    def bfs():
        queue = deque()
        for u in left_nodes:
            if pair_u[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = inf
        found = False
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                pu = pair_v[v]
                if pu is None:
                    found = True
                elif dist[pu] == inf:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in graph[u]:
            pu = pair_v[v]
            if pu is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = inf
        return False

    matching = 0
    while bfs():
        for u in left_nodes:
            if pair_u[u] is None and dfs(u):
                matching += 1
    return matching, pair_u


def assign_people_to_prefixes(candidates_by_prefix: Dict[str, List[str]], seed: int) -> Dict[str, List[str]]:
    person_to_prefixes = build_person_to_prefixes(candidates_by_prefix)
    left_nodes = sorted(person_to_prefixes)
    if len(left_nodes) != 600:
        raise RuntimeError(f"Expected 600 unique person ids, got {len(left_nodes)}")
    rng = random.Random(seed)
    slot_map = {}
    right_nodes = []
    for prefix in PREFIXES:
        slot_names = [f"{prefix}#{idx:03d}" for idx in range(120)]
        rng.shuffle(slot_names)
        slot_map[prefix] = slot_names
        right_nodes.extend(slot_names)
    graph = {}
    shuffled_left = left_nodes[:]
    rng.shuffle(shuffled_left)
    for pid in shuffled_left:
        neighbors = []
        prefixes = person_to_prefixes[pid][:]
        rng.shuffle(prefixes)
        for prefix in prefixes:
            neighbors.extend(slot_map[prefix])
        graph[pid] = neighbors
    matching, pair_u = hopcroft_karp(graph, shuffled_left, right_nodes)
    if matching != 600:
        raise RuntimeError(f"Failed to assign 600 people to prefix slots, only matched {matching}")
    result = defaultdict(list)
    for pid, slot in pair_u.items():
        prefix = slot.split("#", 1)[0]
        result[prefix].append(pid)
    for prefix in PREFIXES:
        result[prefix] = sorted(result[prefix])
        if len(result[prefix]) != 120:
            raise RuntimeError(f"Prefix {prefix} expected 120 people, got {len(result[prefix])}")
    return dict(result)


def split_prefix_people(
    prefix_people: Dict[str, List[str]], seed: int, first_name: str, second_name: str
) -> Dict[str, Dict[str, List[str]]]:
    rng = random.Random(seed)
    result = {}
    for prefix, people in prefix_people.items():
        shuffled = people[:]
        rng.shuffle(shuffled)
        result[prefix] = {
            first_name: sorted(shuffled[:60]),
            second_name: sorted(shuffled[60:120]),
        }
    return result


def save_step1x_npz(
    target_image_path: Path, embeds: np.ndarray, masks: np.ndarray, force_overwrite: bool = False
) -> Path:
    npz_path = target_image_path.with_suffix("")
    npz_path = npz_path.parent / f"{npz_path.name}_step1x_te.npz"
    if npz_path.exists() and not force_overwrite:
        existing = np.load(npz_path)
        if "embeds" in existing and "masks" in existing:
            return npz_path
    np.savez(npz_path, embeds=embeds, masks=masks)
    return npz_path


def build_record(set_id: str, direction: str, prefix: str, person_id: str, pair_id: str) -> Dict[str, str]:
    paths = resolve_image_paths(prefix, person_id, pair_id)
    if set_id == "set1" and direction == "neg_to_pos":
        img1 = paths["flux_neg"]
        img2 = paths["iy_pos"]
        ref_image_path = paths["flux_neg"]
        target_image_path = paths["flux_pos"]
    elif set_id == "set1" and direction == "pos_to_neg":
        img1 = paths["flux_pos"]
        img2 = paths["iy_neg"]
        ref_image_path = paths["flux_pos"]
        target_image_path = paths["flux_neg"]
    elif set_id == "set2" and direction == "pos_to_neg":
        img1 = paths["iy_pos"]
        img2 = paths["flux_neg"]
        ref_image_path = paths["iy_pos"]
        target_image_path = paths["iy_neg"]
    elif set_id == "set2" and direction == "neg_to_pos":
        img1 = paths["iy_neg"]
        img2 = paths["flux_pos"]
        ref_image_path = paths["iy_neg"]
        target_image_path = paths["iy_pos"]
    else:
        raise ValueError((set_id, direction))
    return {
        "set_id": set_id,
        "direction": direction,
        "prefix_family": prefix,
        "person_id": person_id,
        "pair_id": pair_id,
        "img1_path": str(img1.resolve()),
        "img2_path": str(img2.resolve()),
        "ref_image_path": str(ref_image_path.resolve()),
        "target_image_path": str(target_image_path.resolve()),
    }


def expand_split_to_records(split: Dict[str, Dict[str, List[str]]], set_id: str) -> List[Dict[str, str]]:
    records = []
    for prefix in PREFIXES:
        for direction, people in split[prefix].items():
            for person_id in people:
                for pair_id in PAIRS:
                    records.append(build_record(set_id, direction, prefix, person_id, pair_id))
    return records


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(records: List[Dict[str, str]], force_overwrite_npz: bool = False) -> Tuple[dict, List[dict]]:
    metadata = {}
    audit_rows = []
    for record in tqdm(records):
        img1 = Path(record["img1_path"])
        img2 = Path(record["img2_path"])
        ref_image_path = Path(record["ref_image_path"])
        target_image_path = Path(record["target_image_path"])

        mirrored_img1_path = ensure_mirror_file(img1)
        mirrored_img2_path = ensure_mirror_file(img2)
        mirrored_ref_image_path = ensure_mirror_file(ref_image_path)
        mirrored_target_image_path = ensure_mirror_file(target_image_path)

        prompt = build_dual_image_prompt(img1, img2)
        embeds, masks = build_prompt_embedding(img1, prompt)
        npz_path = save_step1x_npz(mirrored_target_image_path, embeds, masks, force_overwrite=force_overwrite_npz)
        key = str(mirrored_target_image_path.resolve())
        if key in metadata:
            raise RuntimeError(f"Duplicate target image key: {key}")
        metadata[key] = {
            "ref_image_path": str(mirrored_ref_image_path.resolve()),
            "caption": prompt,
        }
        audit_row = dict(record)
        audit_row["caption"] = prompt
        audit_row["original_img1_path"] = str(img1.resolve())
        audit_row["original_img2_path"] = str(img2.resolve())
        audit_row["original_ref_image_path"] = str(ref_image_path.resolve())
        audit_row["original_target_image_path"] = str(target_image_path.resolve())
        audit_row["mirrored_img1_path"] = str(mirrored_img1_path.resolve())
        audit_row["mirrored_img2_path"] = str(mirrored_img2_path.resolve())
        audit_row["mirrored_ref_image_path"] = str(mirrored_ref_image_path.resolve())
        audit_row["mirrored_target_image_path"] = str(mirrored_target_image_path.resolve())
        audit_row["embedding_npz_path"] = str(npz_path.resolve())
        audit_rows.append(audit_row)
    return metadata, audit_rows


def build_splits():
    candidates_by_prefix = scan_candidates()
    for prefix in PREFIXES:
        print(prefix, len(candidates_by_prefix[prefix]))
    set1_prefix_people = assign_people_to_prefixes(candidates_by_prefix, SET1_SEED)
    set2_prefix_people = assign_people_to_prefixes(candidates_by_prefix, SET2_SEED)
    set1_split = split_prefix_people(set1_prefix_people, SET1_SEED + 100, "neg_to_pos", "pos_to_neg")
    set2_split = split_prefix_people(set2_prefix_people, SET2_SEED + 100, "pos_to_neg", "neg_to_pos")
    return set1_split, set2_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-set1", action="store_true")
    parser.add_argument("--run-set2", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-single-preview", action="store_true")
    parser.add_argument("--force-overwrite-npz", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_single_preview:
        preview = run_single_preview()
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    set1_split, set2_split = build_splits()
    set1_records = expand_split_to_records(set1_split, "set1")
    set2_records = expand_split_to_records(set2_split, "set2")
    assert len(set1_records) == 3000
    assert len(set2_records) == 3000
    all_records = set1_records + set2_records
    assert len(all_records) == 6000

    split_output = OUTPUT_METADATA_DIR / "connector_6000_split_v1.json"
    set1_output = OUTPUT_METADATA_DIR / "connector_set1_flux2_3000_v1.json"
    set2_output = OUTPUT_METADATA_DIR / "connector_set2_infiniteyou_3000_v1.json"
    combined_output = OUTPUT_METADATA_DIR / "connector_6000_train_v1.json"
    audit_output = OUTPUT_METADATA_DIR / "connector_6000_audit_v1.jsonl"

    split_payload = {
        "master_seed": MASTER_SEED,
        "set1_seed": SET1_SEED,
        "set2_seed": SET2_SEED,
        "prefixes": PREFIXES,
        "pairs": PAIRS,
        "set1": set1_split,
        "set2": set2_split,
    }
    write_json(split_output, split_payload)
    print("Prepared split payload at", split_output)

    run_all = args.run_all or (not args.run_set1 and not args.run_set2)

    if run_all or args.run_set1:
        set1_metadata, set1_audit = generate_dataset(set1_records, force_overwrite_npz=args.force_overwrite_npz)
        write_json(set1_output, set1_metadata)
        print("set1 samples =", len(set1_metadata))
        print("Saved:", set1_output)
    else:
        set1_metadata, set1_audit = {}, []

    if run_all or args.run_set2:
        set2_metadata, set2_audit = generate_dataset(set2_records, force_overwrite_npz=args.force_overwrite_npz)
        write_json(set2_output, set2_metadata)
        print("set2 samples =", len(set2_metadata))
        print("Saved:", set2_output)
    else:
        set2_metadata, set2_audit = {}, []

    if run_all:
        combined_metadata = dict(set1_metadata)
        overlap = set(combined_metadata).intersection(set2_metadata)
        if overlap:
            raise RuntimeError(f"Unexpected overlap between set1 and set2 target keys: {len(overlap)}")
        combined_metadata.update(set2_metadata)
        write_json(combined_output, combined_metadata)
        write_jsonl(audit_output, set1_audit + set2_audit)
        print("combined samples =", len(combined_metadata))
        print("Saved:", combined_output)
        print("Saved:", audit_output)


if __name__ == "__main__":
    main()
