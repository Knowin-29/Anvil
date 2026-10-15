#!/bin/bash
cd ${GSA_FORGE_ROOT}
for f in eval/vlabel_out_*.jsonl; do
  tag=$(basename $f | sed 's/vlabel_out_//' | sed 's/.jsonl//')
  score_f="eval/vlabel_score_${tag}.json"
  if [ ! -f "$score_f" ] && [ -f "$f" ]; then
    echo "Scoring $tag ..."
    python3 eval/score_vlabel.py "$f" eval/vlabel_gt.jsonl "$tag"
  fi
done
echo "Generating HTML..."
python3 eval/gen_vlabel_html.py
echo "Done."
