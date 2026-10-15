#!/usr/bin/env python3
"""Convert HF refcoco parquet to ms-swift grounding format."""
import json, os, io
from pathlib import Path

try:
    import pandas as pd
    from PIL import Image
except ImportError:
    print("Installing deps...")
    os.system("pip install pandas pyarrow pillow -q")
    import pandas as pd
    from PIL import Image

REFCOCO_DIR = "${HF_DOWNLOAD_ROOT}/datasets/refcoco"
OUT_FILE = "${GSA_FORGE_ROOT}/trains/refcoco_grounding.jsonl"
IMG_DIR = "${HF_DOWNLOAD_ROOT}/datasets/refcoco/images"
os.makedirs(IMG_DIR, exist_ok=True)

print("Loading refcoco train parquets...")
parquets = sorted(Path(REFCOCO_DIR, "data").glob("train-*.parquet"))
print(f"Found {len(parquets)} train shards")

total = 0; kept = 0; img_saved = 0
with open(OUT_FILE, 'w') as fout:
    for pf in parquets:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            total += 1
            question = row.get('question', '')
            bbox_str = row.get('bbox', '')
            img_data = row.get('image', None)
            img_size = row.get('image_size', None)
            
            if not question or not bbox_str:
                continue
            
            # Parse bbox "[x1, y1, x2, y2]"
            try:
                bbox = json.loads(bbox_str)
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
            except:
                continue
            
            # Save image if it's bytes/dict
            img_path = os.path.join(IMG_DIR, f"refcoco_{row.get('question_id', total)}.jpg")
            if not os.path.exists(img_path):
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img_bytes = img_data['bytes']
                    with open(img_path, 'wb') as imgf:
                        imgf.write(img_bytes)
                    img_saved += 1
                elif isinstance(img_data, bytes):
                    with open(img_path, 'wb') as imgf:
                        imgf.write(img_data)
                    img_saved += 1
                else:
                    continue
            
            # ms-swift grounding format
            record = {
                "messages": [
                    {"role": "user", "content": "<image>Find the <ref-object> in the image"},
                    {"role": "assistant", "content": "<bbox>"}
                ],
                "images": [img_path],
                "objects": {
                    "ref": [question],
                    "bbox": [bbox]
                }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1
            
            if kept % 10000 == 0:
                print(f"  {kept} converted...", flush=True)

print(f"\nDone: total={total} kept={kept} images_saved={img_saved}")
print(f"Output: {OUT_FILE}")
