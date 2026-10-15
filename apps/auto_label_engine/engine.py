import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import spacy
from PIL import Image, ImageDraw, ImageFont


def setup_spacy(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except OSError:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model], check=True,
        )
        return spacy.load(model)


def extract_noun_phrases(caption: str, nlp) -> List[Tuple[str, int, int]]:
    doc = nlp(caption)
    return [
        (str(chunk).lower().strip(), chunk.start_char, chunk.end_char)
        for chunk in doc.noun_chunks
    ]


def _unique(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _match_phrase_to_prediction(
    phrase: str, grounding_results: Dict[str, Any],
) -> List[Dict]:
    """Try exact match first, then case-insensitive match."""
    if phrase in grounding_results:
        return grounding_results[phrase]
    for key, val in grounding_results.items():
        if key.lower() == phrase.lower():
            return val
    return []


def create_annotation(
    image_path: str,
    caption: str,
    image_size: Tuple[int, int],
    phrases_with_spans: List[Tuple[str, int, int]],
    grounding_results: Dict[str, Any],
) -> Dict:
    width, height = image_size
    annotation: Dict[str, Any] = {
        "image": image_path,
        "caption": caption,
        "width": width,
        "height": height,
        "phrases": [],
    }
    for phrase, start_char, end_char in phrases_with_spans:
        boxes = []
        for det in _match_phrase_to_prediction(phrase, grounding_results):
            if det.get("type") == "box":
                boxes.append({"bbox": det["coords"]})
        if boxes:
            annotation["phrases"].append(
                {
                    "phrase": phrase,
                    "start_char": start_char,
                    "end_char": end_char,
                    "boxes": boxes,
                }
            )
    return annotation


def visualize_grounding(
    image: Image.Image, annotation: Dict, save_path: str,
) -> None:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()

    import matplotlib.pyplot as plt
    colors = plt.cm.tab20(range(20))
    for idx, phrase_data in enumerate(annotation["phrases"]):
        color = tuple((colors[idx % 20][:3] * 255).astype(int).tolist())
        for box_data in phrase_data["boxes"]:
            x1, y1, x2, y2 = box_data["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 18), phrase_data["phrase"], fill=color, font=font)
    vis.save(save_path)


def run_grounding_batch(
    image_caption_pairs: List[Tuple[str, str]],
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
    output_jsonl: str = "data/output/grounding_annotations.jsonl",
    output_dir: str = "data/output/grounding_visualizations",
    visualize: bool = True,
) -> Dict:
    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if visualize:
        os.makedirs(output_dir, exist_ok=True)

    nlp = setup_spacy()
    from rex_omni import RexOmniWrapper
    model = RexOmniWrapper(
        model_path=model_path,
        backend=backend,
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    annotations: List[Dict] = []

    for image_path, caption in image_caption_pairs:
        image = Image.open(image_path).convert("RGB")
        phrases_with_spans = extract_noun_phrases(caption, nlp)
        phrases = _unique([p[0] for p in phrases_with_spans])
        if not phrases:
            continue
        result = model.inference(
            images=image, task="detection", categories=phrases
        )[0]
        annotation = create_annotation(
            image_path=image_path,
            caption=caption,
            image_size=image.size,
            phrases_with_spans=phrases_with_spans,
            grounding_results=result.get("extracted_predictions", {}),
        )
        if annotation["phrases"]:
            annotations.append(annotation)
            if visualize:
                name = os.path.splitext(os.path.basename(image_path))[0]
                vis_path = os.path.join(output_dir, f"{name}_grounding.jpg")
                visualize_grounding(image, annotation, vis_path)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "count": len(annotations),
        "output_jsonl": output_jsonl,
        "output_dir": output_dir if visualize else None,
    }
