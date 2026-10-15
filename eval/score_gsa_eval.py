#!/usr/bin/env python3
"""Score per-dimension GSA eval outputs against thousandth-scale GT.

Metrics per dimension:
  G_grounding          : parse_rate, mean_iou, iou@0.5, iou@0.7
  S_spatial            : predicate_acc (exact, normalized), triple_acc (subj+pred+obj)
  A_identity_attr      : attr_f1 (key:value overlap F1 vs GT attributes)
  A_obj_state          : state_value_acc (normalized exact/substring)
  A_action_precondition: can_do_now_acc
  A_action_interface   : point_in_bbox (grasp point inside GT bbox), mean_norm_dist

Usage: python eval/score_gsa_eval.py --dim <dim> --pred <out.jsonl> --gt <gt.jsonl> --name X
"""
import argparse
import json
import re
import statistics as st


def get_resp(o):
    if "response" in o:
        return str(o["response"])
    if o.get("messages"):
        return str(o["messages"][-1].get("content", ""))
    return ""


def parse_json_obj(text):
    """Extract the last {...} JSON object from text (handles <think> + markdown fences)."""
    text = re.sub(r"```(?:json)?", "", str(text))
    objs = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for cand in reversed(objs):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return {}


def last_bbox(text):
    m = re.findall(r'(?:bbox|bbox_2d)"?\s*:?\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]', str(text))
    if m:
        return [float(v) for v in m[-1]]
    m = re.findall(r'\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]', str(text))
    return [float(v) for v in m[-1]] if m else None


def last_point(text):
    m = re.findall(r'(?:grasp_point_2d|point)"?\s*:?\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]', str(text))
    if m:
        return [float(v) for v in m[-1]]
    m = re.findall(r'\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]', str(text))
    return [float(v) for v in m[-1]] if m else None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def iou(a, b):
    if not a or not b:
        return 0.0
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    if xa1 > xa2: xa1, xa2 = xa2, xa1
    if ya1 > ya2: ya1, ya2 = ya2, ya1
    ix1, iy1, ix2, iy2 = max(xa1, xb1), max(ya1, yb1), min(xa2, xb2), min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / ua if ua > 0 else 0.0


def score(dim, preds, gts):
    n = min(len(preds), len(gts))
    r = {"dim": dim, "n": n}
    if dim == "G_grounding":
        ious, ok = [], 0
        for o, g in zip(preds[:n], gts[:n]):
            p = last_bbox(get_resp(o))
            if p: ok += 1
            ious.append(iou(p, [float(x) for x in g["bbox"]]) if p else 0.0)
        r.update(parse_rate=round(100*ok/n, 1), mean_iou=round(st.mean(ious), 3),
                 **{"iou@0.5": round(100*sum(i >= .5 for i in ious)/n, 1),
                    "iou@0.7": round(100*sum(i >= .7 for i in ious)/n, 1)})
    elif dim == "S_spatial":
        pred_ok = trip_ok = 0
        for o, g in zip(preds[:n], gts[:n]):
            d = parse_json_obj(get_resp(o))
            if norm(d.get("predicate")) == norm(g["predicate"]): pred_ok += 1
            if (norm(d.get("predicate")) == norm(g["predicate"]) and
                    norm(d.get("subject")) == norm(g["subject"]) and
                    norm(d.get("object")) == norm(g["object"])): trip_ok += 1
        r.update(predicate_acc=round(100*pred_ok/n, 1), triple_acc=round(100*trip_ok/n, 1))
    elif dim == "A_identity_attr":
        f1s = []
        for o, g in zip(preds[:n], gts[:n]):
            d = parse_json_obj(get_resp(o))
            pa = d.get("attributes", {}) if isinstance(d.get("attributes"), dict) else {}
            ga = g["attributes"] if isinstance(g["attributes"], dict) else {}
            pset = {f"{norm(k)}:{norm(v)}" for k, v in pa.items()}
            gset = {f"{norm(k)}:{norm(v)}" for k, v in ga.items()}
            if not gset:
                continue
            tp = len(pset & gset)
            prec = tp/len(pset) if pset else 0
            rec = tp/len(gset) if gset else 0
            f1s.append(2*prec*rec/(prec+rec) if (prec+rec) else 0.0)
        r.update(attr_f1=round(st.mean(f1s), 3) if f1s else 0.0, scored=len(f1s))
    elif dim == "A_obj_state":
        ok = 0
        for o, g in zip(preds[:n], gts[:n]):
            d = parse_json_obj(get_resp(o))
            gv, pv = norm(g["state_value"]), norm(d.get("state_value"))
            if gv and (gv == pv or gv in pv or pv in gv): ok += 1
        r.update(state_value_acc=round(100*ok/n, 1))
    elif dim == "A_action_precondition":
        ok = 0
        for o, g in zip(preds[:n], gts[:n]):
            d = parse_json_obj(get_resp(o))
            if isinstance(d.get("can_do_now"), bool) and d["can_do_now"] == g["can_do_now"]: ok += 1
        r.update(can_do_now_acc=round(100*ok/n, 1))
    elif dim == "A_action_interface":
        inb, dists, ok = 0, [], 0
        for o, g in zip(preds[:n], gts[:n]):
            p = last_point(get_resp(o))
            if p: ok += 1
            gb = g.get("bbox")
            if p and gb and gb[0] <= p[0] <= gb[2] and gb[1] <= p[1] <= gb[3]:
                inb += 1
            if p:
                gp = g["grasp_point_2d"]
                dists.append(((p[0]-gp[0])**2 + (p[1]-gp[1])**2)**0.5 / 1000.0)
        r.update(parse_rate=round(100*ok/n, 1),
                 point_in_bbox=round(100*inb/n, 1),
                 mean_norm_dist=round(st.mean(dists), 3) if dists else None)
    print(json.dumps(r, ensure_ascii=False))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--name", default="model")
    args = ap.parse_args()
    preds = [json.loads(l) for l in open(args.pred)]
    gts = [json.loads(l) for l in open(args.gt)]
    score(args.dim, preds, gts)


if __name__ == "__main__":
    main()
