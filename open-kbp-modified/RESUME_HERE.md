# Resume after pod pause — SmoothAdv run

State when paused (2026-08-10 ~03:55 UTC):

* All 3 models TRAINED, on the volume at /workspace/results/*smoothadv_s{0.02,0.05,0.10}*/models/epoch_100.keras
* sigma=0.02 CERTIFIED — result committed, see reports/certified_robustness/experiment_results/certify_smoothadv_s0.02/
* sigma=0.05, sigma=0.10 NOT certified yet (0.05 was ~2 min in when paused)
* 5 local commits ahead of 9b0a64d; pod has NO push credential by design

## Steps to resume

```bash
# 1. dataset lives on the container overlay and is wiped by a pause — re-clone (~1 min)
git clone https://github.com/ababier/open-kbp.git /tmp/okbp-data

# 2. relaunch the certification driver (detached, survives session teardown)
cd /workspace/openkbp/open-kbp-modified
setsid nohup ./certify_all.sh > certify_driver.log 2>&1 < /dev/null &
disown
ps -eo pid,ppid,cmd | grep "bash ./certify_all.sh" | grep -v grep   # PPID must be 1

# 3. when each sigma lands
python compare_certs.py 0.02 0.05 0.10
```

certify_all.sh skips nothing silently — it exits 1 with "CERTIFICATION INCOMPLETE"
if any sigma is missing. It re-certifies sigma=0.02 too; delete or move
certify_smoothadv_s0.02/ first if you want to skip it (the committed copy is
already under reports/).

## Gotchas that cost time already

* Models are at the ABSOLUTE path /workspace/results, NOT ./results (runpod_train.py:147).
* --batch-draws 32 OOMs on the 4090; 16 works at ~61 s/patient, 40 patients => ~41 min/sigma.
* /workspace is only 20 GB and checkpoints are up to 1.79 GB each — prune all but
  epoch_100.keras per model or a later run dies mid-write at ~93% full.
* Claude memory files were on the container overlay; a copy is at
  /workspace/claude-memory-backup/ — restore to /root/.claude/projects/-/memory/.

## sigma=0.02 result (already obtained)

  radius | D95 PTV70 width (base -> smoothadv) | mean Brainstem  | frac<=1Gy
   0.01  | 0.4538 -> 0.4833  (+6.5%)           | -27.1%          | -0.7%
   0.02  | 0.7553 -> 0.8070  (+6.8%)           | -27.3%          | -5.9%
   0.028 | 1.1154 -> 1.1935  (+7.0%)           | -28.5%          | -5.2%

MIXED, not a uniform win: PTV D95 widened, brainstem tightened a lot, frac<=1Gy fell.
