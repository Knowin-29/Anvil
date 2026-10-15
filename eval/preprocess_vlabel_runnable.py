#!/usr/bin/env python3
"""Make vlabel rows runnable by compressing long text and oversized images.

This keeps every valid sample instead of skipping long-text or huge-image rows.
Only genuinely missing/corrupt images are skipped.
"""
import argparse
import hashlib
import json
import os
import re

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compress_text(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text, False
    image_prefix = ""
    if text.startswith("<image>"):
        image_prefix = "<image>\n"
        text = text[len("<image>"):].lstrip()

    template = ""
    m = re.search(r"<OUTPUT TEMPLATE>.*?</OUTPUT TEMPLATE>", text, re.S)
    if m:
        template = m.group(0).strip() + "\n"

    budget = max_chars - len(image_prefix) - len(template) - 120
    budget = max(1200, budget)
    head = text[: budget // 2].strip()
    tail = text[-budget // 2 :].strip()
    compressed = (
        f"{image_prefix}{template}{head}\n\n"
        "[... compressed long prompt for evaluation stability ...]\n\n"
        f"{tail}"
    )
    return compressed[:max_chars], True


def compress_image(path, cache_dir, max_side, quality):
    if not os.path.isfile(path):
        return None, "missing_image", None
    if Image is None:
        return path, None, None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            old_size = [im.width, im.height]
            if max(im.width, im.height) <= max_side:
                return path, None, {"old_size": old_size, "new_size": old_size, "compressed": False}
            os.makedirs(cache_dir, exist_ok=True)
            key = hashlib.sha1(f"{path}:{os.path.getmtime(path)}:{max_side}".encode()).hexdigest()[:20]
            out = os.path.join(cache_dir, f"{key}.jpg")
            if not os.path.exists(out):
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                im.save(out, "JPEG", quality=quality, optimize=True)
            with Image.open(out) as nim:
                new_size = [nim.width, nim.height]
            return out, None, {"old_path": path, "new_path": out, "old_size": old_size, "new_size": new_size, "compressed": True}
    except Exception as exc:
        return None, "bad_image", {"path": path, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-input", required=True)
    ap.add_argument("--out-gt", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--image-cache", default="${GSA_EVAL_ROOT}/compressed_images")
    ap.add_argument("--max-text-chars", type=int, default=7000)
    ap.add_argument("--max-image-side", type=int, default=1344)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    inputs = read_jsonl(args.input)
    gts = read_jsonl(args.gt)
    report = {
        "input_n": len(inputs),
        "gt_n": len(gts),
        "kept": 0,
        "skipped": 0,
        "text_compressed": [],
        "image_compressed": [],
        "skip_rows": [],
        "config": {
            "max_text_chars": args.max_text_chars,
            "max_image_side": args.max_image_side,
            "image_cache": args.image_cache,
        },
    }

    with open(args.out_input, "w", encoding="utf-8") as fi, open(args.out_gt, "w", encoding="utf-8") as fg:
        for local_i, (inp0, gt0) in enumerate(zip(inputs, gts)):
            inp = json.loads(json.dumps(inp0, ensure_ascii=False))
            gt = dict(gt0)
            gt.setdefault("original_index", local_i)

            text_changed = False
            for msg in inp.get("messages") or []:
                content, changed = compress_text(msg.get("content", ""), args.max_text_chars)
                msg["content"] = content
                text_changed = text_changed or changed
            if text_changed:
                report["text_compressed"].append({
                    "local_index": local_i,
                    "original_index": gt.get("original_index", local_i),
                    "id": gt.get("id"),
                    "task_family": gt.get("task_family"),
                })

            new_images = []
            skip_reasons = []
            for img in inp.get("images") or []:
                new_img, reason, meta = compress_image(img, args.image_cache, args.max_image_side, args.jpeg_quality)
                if reason:
                    skip_reasons.append(reason)
                    if meta:
                        skip_reasons.append(meta.get("error", ""))
                    continue
                new_images.append(new_img)
                if meta and meta.get("compressed"):
                    report["image_compressed"].append({
                        "local_index": local_i,
                        "original_index": gt.get("original_index", local_i),
                        "id": gt.get("id"),
                        "task_family": gt.get("task_family"),
                        **meta,
                    })
            if not new_images:
                skip_reasons.append("no_runnable_image")
            if skip_reasons:
                report["skipped"] += 1
                report["skip_rows"].append({
                    "local_index": local_i,
                    "original_index": gt.get("original_index", local_i),
                    "id": gt.get("id"),
                    "task_family": gt.get("task_family"),
                    "reasons": sorted(set(x for x in skip_reasons if x)),
                    "images": inp0.get("images", []),
                })
                continue
            inp["images"] = new_images
            fi.write(json.dumps(inp, ensure_ascii=False) + "\n")
            fg.write(json.dumps(gt, ensure_ascii=False) + "\n")
            report["kept"] += 1

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        f"Preprocessed rows: kept={report['kept']} skipped={report['skipped']} "
        f"text_compressed={len(report['text_compressed'])} image_compressed={len(report['image_compressed'])}"
    )
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
