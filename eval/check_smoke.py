import json, re
def parse_bbox(text):
    m = re.search(r'"bbox"\s*:\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', text)
    if m: return [float(m.group(i)) for i in range(1,5)]
    m = re.search(r'<box>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</box>', text)
    if m: return [float(m.group(i)) for i in range(1,5)]
    return None

def iou(a, b):
    if not a or not b: return None
    xa1,ya1,xa2,ya2 = a; xb1,yb1,xb2,yb2 = b
    ix1,iy1 = max(xa1,xb1), max(ya1,yb1)
    ix2,iy2 = min(xa2,xb2), min(ya2,yb2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    ua = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
    return inter/ua if ua>0 else 0

outs = [json.loads(l) for l in open('eval/replay_smoke_out.jsonl')]
gts = [json.loads(l) for l in open('eval/replay_gt.jsonl')][:10]

for i, (o, g) in enumerate(zip(outs, gts)):
    resp = o.get('response','')
    pred = parse_bbox(str(resp))
    gtb = g['gt_bbox']
    iouv = iou(pred, gtb)
    print(f"[{i}] dim={g['dimension']}")
    print(f"  PRED: {str(resp)[:90]}")
    print(f"  GT  : {g['gt_response'][:90]}")
    print(f"  pred_bbox={pred} gt={gtb} IoU={iouv:.3f}" if iouv is not None else f"  pred_bbox={pred} gt={gtb} IoU=N/A")
    print()
