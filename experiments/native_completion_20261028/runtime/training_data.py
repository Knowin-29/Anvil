"""Source-preserving native SFT encoding for the isolated HF training runtime."""
import json
import math
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

BASE = "/Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def processor():
    p = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    p.image_processor.size = {"longest_edge": 1536 * 32 * 32, "shortest_edge": 4 * 32 * 32}
    return p


def encode(p, row, max_length=16384):
    assert row["gsa_meta"]["split"] == "train"
    messages = row["messages"]
    assert len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
    question = messages[0]["content"]
    assert question.startswith("<image>") and question.count("<image>") == 1
    chat = [{"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": question.removeprefix("<image>")}]}]
    rendered = p.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert rendered.endswith(EMPTY_THINK)
    target = EMPTY_THINK + messages[1]["content"] + "<|im_end|>\n"
    full = rendered.removesuffix(EMPTY_THINK) + target
    with Image.open(row["images"][0]) as im:
        im = im.convert("RGB")
        # Match qwen-vl-utils' PIL resize used by the original Swift pipeline.
        # Resizing with torchvision inside the processor otherwise differs by
        # one uint8 intensity step on some pixels.
        width, height = im.size
        assert max(width, height) / min(width, height) <= 200
        h = max(32, round(height / 32) * 32)
        w = max(32, round(width / 32) * 32)
        if h * w > 1536 * 32 * 32:
            scale = math.sqrt(height * width / (1536 * 32 * 32))
            h, w = math.floor(height / scale / 32) * 32, math.floor(width / scale / 32) * 32
        elif h * w < 4 * 32 * 32:
            scale = math.sqrt(4 * 32 * 32 / (height * width))
            h, w = math.ceil(height * scale / 32) * 32, math.ceil(width * scale / 32) * 32
        im = im.resize((w, h))
        data = p(text=[full], images=[im], do_resize=False, return_tensors="pt")
    tokens = data["input_ids"]
    target_ids = p.tokenizer.encode(target, add_special_tokens=False)
    assert tokens.shape[-1] <= max_length, f"Sequence exceeds context; refusing to drop or clip: {row['id']}"
    assert tokens[0, -len(target_ids):].tolist() == target_ids
    labels = tokens.clone()
    labels[:, :-len(target_ids)] = -100
    data["labels"] = labels
    return dict(data)


def compare_original_fixtures():
    root = Path(__file__).resolve().parent / "training_repair/encoding_probe"
    p = processor()
    result = []
    for meta in json.loads((root / "report.json").read_text()):
        form = meta["form"]
        row = json.loads((root / (form + ".json")).read_text())
        old = torch.load(root / (form + ".pt"), weights_only=True)
        new = encode(p, row)
        entry = {"form": form}
        for key in ("input_ids", "labels", "image_grid_thw", "pixel_values"):
            a, b = old[key], new[key]
            entry[key] = dict(old_shape=list(a.shape), new_shape=list(b.shape),
                              exact=torch.equal(a, b))
            if a.shape == b.shape:
                entry[key]["max_abs_diff"] = float((a.float() - b.float()).abs().max())
        assert entry["input_ids"]["exact"] and entry["labels"]["exact"], entry
        result.append(entry)
    dest = root.parent / "encoding_comparison.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    compare_original_fixtures()
