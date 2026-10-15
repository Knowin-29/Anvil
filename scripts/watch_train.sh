#!/usr/bin/env bash
# One-shot health summary for the newest (or a named) training log.
#   bash scripts/watch_train.sh [logfile]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LOG="${1:-$(ls -t logs/sft_*_dlc_*.log logs/9b_*.log 2>/dev/null | head -1)}"
[[ -s "$LOG" ]] || { echo "no training log found"; exit 1; }

python3 - "$LOG" <<'PY'
import re, sys, os, time
p = sys.argv[1]
t = open(p, errors='ignore').read()

# where did this run? host RAM distinguishes the DLC node (~2TB) from the dev box (64G)
# Distinguish DLC from the dev box. `host RAM` is only present when our wrapper
# printed the header; if the log starts mid-run (platform captured stdout
# separately) fall back to the output_dir version stamp, which the dev box never
# writes because it uses a different run name.
ram = re.search(r'host RAM: (\d+)G', t)
if ram:
    where = 'DLC (remote)' if int(ram.group(1)) > 200 else 'local dev box'
elif re.search(r'output_dir=/path/to/workspace/demo/\S+/output/sft_9b_gsabench', t):
    where = 'DLC (remote, header not captured)'
else:
    where = 'unknown'
ng = re.search(r'GPUs\s*[:=]\s*(\d+)', t)
tuner = re.search(r'tuner_type=(\w+)', t) or re.search(r'--tuner_type (\w+)', t)
rank = re.search(r'lora_rank=(\d+)', t)

steps = {}
for m in re.finditer(r"'loss': '([\d.]+)'.*?'token_acc': '([\d.]+)'"
                     r".*?'global_step/max_steps': '(\d+)/(\d+)'", t):
    steps[int(m.group(3))] = (float(m.group(1)), float(m.group(2)), int(m.group(4)))
ev = re.findall(r"'eval_loss': '([\d.]+)'.*?'eval_token_acc': '([\d.]+)'"
                r".*?'global_step/max_steps': '(\d+)/", t)
sp = re.findall(r"'train_speed\(s/it\)': '([\d.]+)'", t)
el = re.findall(r"'elapsed_time': '([^']+)'", t)
mem = re.findall(r"'memory\(GiB\)': '([\d.]+)'", t)

print(f"log   : {os.path.basename(p)}")
print(f"where : {where}" + (f"   GPUs={ng.group(1)}" if ng else ""))
if tuner:
    print(f"tuner : {tuner.group(1)}" + (f" r{rank.group(1)}" if rank else ""))
age = (time.time() - os.path.getmtime(p)) / 60
print(f"updated: {age:.0f} min ago" + ("   <-- STALLED?" if age > 30 else ""))

if not steps:
    print("\nno training steps logged yet")
    raise SystemExit(0)

ks = sorted(steps); cur, tot = ks[-1], steps[ks[-1]][2]
print(f"\nprogress: {cur}/{tot} ({cur/tot*100:.0f}%)")
print(f"loss    : {steps[ks[0]][0]:.4f} -> {steps[cur][0]:.4f}")
print(f"token_acc: {steps[ks[0]][1]:.4f} -> {steps[cur][1]:.4f}")
if ev:
    print("eval_loss: " + " -> ".join(a for a, _, _ in ev) +
          ("  (falling, healthy)" if len(ev) > 1 and float(ev[-1][0]) < float(ev[0][0])
           else "  <-- NOT falling, check overfitting"))
if sp:  print(f"speed   : {sp[-1]} s/step")
if mem: print(f"memory  : {mem[-1]} GiB/GPU")
if el:  print(f"elapsed : {el[-1]}")
if sp and cur < tot:
    rem = (tot - cur) * float(sp[-1]) / 3600
    print(f"ETA     : ~{rem:.1f}h remaining")

# health verdict
issues = []
if steps[cur][1] < 0.82: issues.append(f"token_acc {steps[cur][1]:.3f} below 0.82")
if steps[cur][0] > 0.7:  issues.append(f"loss {steps[cur][0]:.3f} above 0.7")
if mem and float(mem[-1]) > 75: issues.append(f"memory {mem[-1]} GiB near limit")
if len(ev) > 1 and float(ev[-1][0]) > float(ev[-2][0]): issues.append("eval_loss rose")
print("\nverdict : " + ("HEALTHY" if not issues else "CHECK: " + "; ".join(issues)))
PY
