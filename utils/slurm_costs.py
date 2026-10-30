#!/usr/bin/env python3
"""Per-method compute-cost accounting from SLURM's own records (sacct).

The per-dataset baselines (esp. fingerprint's per-cell susie eval) are enormously
more expensive than a single ExPert forward pass — this quantifies that so the cost
can be reported ALONGSIDE the accuracy in the benchmark. For every method it sums,
split by TRAIN vs EVAL:
  - n_jobs, n distinct nodes
  - CPU-hours   = sum(elapsed_sec * alloc_cpus) / 3600
  - GPU-hours   = sum(elapsed_sec * n_gpu)      / 3600
  - node-hours  = sum(elapsed_sec * n_nodes)    / 3600
  - max RAM (GB) across jobs (MaxRSS, from the .batch step)
  - wall-clock span (first start -> last end)

Usage:
  python utils/slurm_costs.py --since 2026-07-01 [--until 2026-07-05] [--out costs.csv]
  # restrict to one run window; states default to all terminal + running.

SLURM keeps this per job, so no instrumentation is needed — we just read it back.
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

# snakemake job names look like "<rule>_size=<sz>.fold=<f>...."; map <rule> to
# (method, stage). Rules absent here are ignored (controllers, unrelated jobs).
RULE_MAP = {
    # fingerprint (per-dataset)
    "fp_pds_prep": ("fingerprint", "train"), "fp_pds_deg_chunk": ("fingerprint", "train"),
    "fp_pds_fit_chunk": ("fingerprint", "train"), "fp_pds_concat_deg": ("fingerprint", "train"),
    "fp_pds_concat_dict": ("fingerprint", "train"),
    "fp_ens_eval_ds_chunk": ("fingerprint", "eval"), "fp_ens_concat": ("fingerprint", "eval"),
    # merged (combined-weights) fingerprint — a DISTINCT method for cost accounting
    "fp_merge_eval_chunk": ("fingerprint_merged", "eval"), "fp_merge_concat": ("fingerprint_merged", "eval"),
    # cellot
    "cellot_ae_pds": ("cellot", "train"), "cellot_pert_chunk": ("cellot", "train"),
    "cellot_ens_eval": ("cellot", "eval"),
    # gears
    "gears_prep_pds": ("gears", "train"), "gears_train_pds": ("gears", "train"),
    "gears_ens_eval": ("gears", "eval"),
    # scouter
    "scouter_train_pds": ("scouter", "train"), "scouter_ens_eval": ("scouter", "eval"),
    # genepert (single job does train + eval), scgpt, baselines
    "genepert": ("genepert", "train+eval"),
    # scgpt: the eval rules are scgpt_eval_chunk (96/run) + scgpt_concat. The old
    # "scgpt_eval" key matched NO rule, so every eval job was silently dropped and
    # scgpt would have reported ~zero eval cost.
    "scgpt_train": ("scgpt", "train"),
    "scgpt_eval_chunk": ("scgpt", "eval"), "scgpt_concat": ("scgpt", "eval"),
    # scGPT is run WITHOUT a controller (utils/scgpt_{train,eval}_manual.sh) because
    # scgpt_train has no upstream DAG dependency and a crashed controller kills its
    # orphans. Those jobs carry the sbatch --job-name, not the snakemake rule name,
    # so without these keys rule_of() yields "scgpt-train-manual" -> unmapped ->
    # every scGPT job silently dropped from the cost table (which is exactly what
    # happened: scGPT was scored in metrics but absent from cost_by_method_stage).
    "scgpt-train-manual": ("scgpt", "train"),
    "scgpt-eval-manual": ("scgpt", "eval"),
    # legacy shared-genes fingerprint (opt-in; not in the default methods list).
    # Mapped to a DISTINCT method so enabling it can never silently merge into the
    # per-dataset fingerprint's numbers -- or get dropped the way scgpt_eval was.
    "fp_prep": ("fingerprint_shared", "train"), "fp_deg_chunk": ("fingerprint_shared", "train"),
    "fp_fit_chunk": ("fingerprint_shared", "train"), "fp_concat_deg": ("fingerprint_shared", "train"),
    "fp_concat_dict": ("fingerprint_shared", "train"),
    "fp_eval_chunk": ("fingerprint_shared", "eval"), "fp_concat_eval": ("fingerprint_shared", "eval"),
    "baselines": ("baselines", "eval"),
    "expert_logits": ("ExPert", "eval"),
    # supervised discriminative baselines: one job trains + scores both splits
    "xgboost": ("xgboost", "train+eval"),
    # the capped twin is its own rule (and its own method in the metrics); without
    # this key every xgboost_capped job was silently dropped from the cost table
    "xgboost_capped": ("xgboost_capped", "train+eval"),
    "linreg": ("linreg", "train+eval"),
}


def rule_of(jobname):
    return jobname.split("_size=")[0].split(".")[0]


# Jobs NOT launched by the snakemake controller carry a hand-written --job-name
# with no `size=` wildcard, so size_of() yields "-" and analysis/cost.py -- which
# filters the CSV to its own size -- drops them entirely. That is how scGPT went
# missing from the cost table even over a full window. Pin their size here.
FORCED_SIZE = {
    "scgpt-train-manual": "small",
    "scgpt-eval-manual": "small",
}


def size_of(jobname):
    """Extract the `size=` wildcard from a job name (e.g. "fp_..._size=full.fold=..."
    -> "full"). "-" when the rule has no size wildcard (rare for accounted rules)."""
    if "_size=" not in jobname:
        return FORCED_SIZE.get(rule_of(jobname), "-")
    return jobname.split("_size=", 1)[1].split(".")[0].split(",")[0]


def cputime_sec(s):
    """sacct TotalCPU ("[DD-]HH:MM:SS[.uuu]" or "MM:SS.uuu") -> seconds. Used for
    CPU utilisation (TotalCPU / allocated core-seconds)."""
    if not s or s in ("None", "Unknown"):
        return 0.0
    days = 0.0
    if "-" in s:
        dd, s = s.split("-", 1)
        try:
            days = float(dd)
        except ValueError:
            return 0.0
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return 0.0
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return days * 86400 + sec


def span_hours(start, end):
    """Wall-clock span between the first Start and last End (ISO), in hours; 0 on
    unparseable timestamps. cpu_hours / wall_hours ~ effective parallelism."""
    from datetime import datetime
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600)
    except (ValueError, TypeError):
        return 0.0


def n_gpu(alloctres):
    for tok in alloctres.split(","):
        if tok.startswith(("gres/gpu=", "gpu=")):
            try:
                return int(tok.split("=")[1])
            except ValueError:
                return 0
    return 0


def rss_gb(maxrss):
    if not maxrss:
        return 0.0
    u = maxrss[-1]
    try:
        v = float(maxrss[:-1]) if u in "KMGT" else float(maxrss)
    except ValueError:
        return 0.0
    return {"K": v / 1e6, "M": v / 1e3, "G": v, "T": v * 1e3}.get(u, v / 1e9)


SACCT_O = ("JobID,JobName%200,ElapsedRaw,AllocCPUS,AllocTRES%60,NNodes,MaxRSS,"
           "NodeList%20,State,Start,End,TotalCPU")


def _new_agg():
    return {"n": 0, "cpuh": 0.0, "gpuh": 0.0, "nodeh": 0.0, "maxram": 0.0,
            "totcpu": 0.0, "nodes": set(), "start": None, "end": None}


def accumulate(agg, sacct_text, want, keyfn, dedupe=False):
    """Fold sacct -P output into agg[key], key = keyfn(jobname) (None -> skip that
    job). want = accepted states (None = all — used for external jobs, where a
    timed-out run that yielded a usable checkpoint must still count). .batch MaxRSS
    + TotalCPU join to the main line by base id.

    dedupe: count each snakemake job ONCE -- its last accepted attempt. A snakemake
    job name carries the rule AND every wildcard ("fp_ens_eval_ds_chunk_size=small.
    fold=fold_0.split=ood_test.dataset=X.j=0002"), so equal names are the same unit
    of work. Without this, re-running a stage after a fix books it again: small's
    fingerprint eval ran three times (Jul 3, Aug 23-24, Aug 27-29) and the table
    reported 19.8k core-h for a 12.0k core-h pass. Hand-named jobs (no `_size=`,
    e.g. scgpt-eval-manual) are never collapsed -- their names are not unique."""
    main_rows, rss, tcpu = {}, {}, {}
    for line in sacct_text.splitlines():
        f = line.split("|")
        if len(f) < 12:
            continue
        jid, name, esec, cpus, tres, nnodes, maxrss, nodelist, state, start, end, totcpu = f[:12]
        base = jid.split(".")[0]
        if "." not in jid:
            state = state.split()[0]                      # "CANCELLED by 123" -> CANCELLED
            if want and state.upper() not in want:
                continue
            main_rows[base] = (name, int(esec or 0), int(cpus or 0), n_gpu(tres),
                               int(nnodes or 0), nodelist, start, end)
        if maxrss:
            rss[base] = max(rss.get(base, 0.0), rss_gb(maxrss))
        if totcpu:
            tcpu[base] = max(tcpu.get(base, 0.0), cputime_sec(totcpu))

    if dedupe:
        last, n_before = {}, len(main_rows)
        for base, row in main_rows.items():
            name, end = row[0], row[7] or ""
            key = name if "_size=" in name else base
            if key not in last or end >= main_rows[last[key]][7]:
                last[key] = base
        main_rows = {b: main_rows[b] for b in last.values()}
        if n_before != len(main_rows):
            print(f"dedupe: {n_before} accepted attempts -> {len(main_rows)} unique jobs "
                  f"(re-runs of the same rule+wildcards counted once)", file=sys.stderr)

    for base, (name, esec, cpus, ng, nn, nodelist, start, end) in main_rows.items():
        k = keyfn(name)
        if k is None:
            continue
        a = agg[k]
        a["n"] += 1
        a["cpuh"] += esec * cpus / 3600
        a["gpuh"] += esec * ng / 3600
        a["nodeh"] += esec * nn / 3600
        a["maxram"] = max(a["maxram"], rss.get(base, 0.0))
        a["totcpu"] += tcpu.get(base, 0.0)
        for nd in nodelist.replace("(None)", "").split(","):
            if nd:
                a["nodes"].add(nd)
        if start and start != "None":
            a["start"] = start if a["start"] is None else min(a["start"], start)
        if end and end not in ("None", "Unknown"):
            a["end"] = end if a["end"] is None else max(a["end"], end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="sacct -S (e.g. 2026-07-01 or 2026-07-01T00:00)")
    ap.add_argument("--until", default=None, help="sacct -E")
    ap.add_argument("--user", default=None, help="default: $USER")
    ap.add_argument("--states", default="COMPLETED",
                    help="comma states to include (default: COMPLETED — cost of the jobs that "
                         "actually produced a result, excluding failed/OOM/cancelled retries). "
                         "Pass --states '' (empty) to count ALL jobs incl. wasted retries.")
    ap.add_argument("--external", action="append", default=[], metavar="JOBID:SIZE:METHOD:STAGE",
                    help="book an out-of-pipeline job (e.g. externally-trained ExPert) by its "
                         "SLURM job id, REGARDLESS of state — a training run that timed out but "
                         "produced a usable best-checkpoint still cost its compute. Repeatable. "
                         "e.g. --external <jobid>:small:ExPert:train (array ids fan in as n_jobs).")
    ap.add_argument("--out", default=None, help="write the per-(size,method,stage) table to this CSV")
    ap.add_argument("--all-passes", action="store_true",
                    help="book every accepted attempt, including re-runs of a job that had "
                         "already completed (default: each rule+wildcards job once, its last "
                         "attempt -- the cost of one clean pass)")
    ap.add_argument("--merge", action="store_true",
                    help="upsert into an existing --out instead of overwriting: this run's "
                         "(size,method,stage) rows replace the matching old ones; methods that "
                         "didn't run this window persist (so cellot/gears/scouter/scgpt from "
                         "earlier sessions — and a hand-added ExPert row — survive). Without "
                         "it, --out is overwritten with only this window's methods.")
    args = ap.parse_args()

    # NOTE: do NOT pass sacct --state alongside -S — it silently drops most jobs
    # (a known sacct quirk). Fetch all jobs in the window and filter states in
    # Python. Default keeps only COMPLETED — the cost of jobs that actually produced
    # a result — so OOM/failed/cancelled retries don't inflate the numbers; --states ''
    # widens it back to every job that consumed resources.
    want = {s.strip().upper() for s in args.states.split(",")} if args.states else None
    agg = defaultdict(_new_agg)

    # 1) pipeline jobs: window pull, keyed by rule -> (size, method, stage).
    cmd = ["sacct", "-u", args.user or os.environ["USER"], "-S", args.since,
           "-P", "--noheader", "-o", SACCT_O]
    if args.until:
        cmd += ["-E", args.until]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout

    def pipeline_key(name):
        rule = rule_of(name)
        return ((size_of(name),) + RULE_MAP[rule]) if rule in RULE_MAP else None

    accumulate(agg, out, want, pipeline_key, dedupe=not args.all_passes)

    # 2) external jobs (e.g. ExPert): fetch each by id, ALL states, fixed key.
    for spec in args.external:
        try:
            jid, size, method, stage = spec.split(":")
        except ValueError:
            print(f"skip --external '{spec}': expected JOBID:SIZE:METHOD:STAGE", file=sys.stderr)
            continue
        ext = subprocess.run(["sacct", "-j", jid, "-P", "--noheader", "-o", SACCT_O],
                             capture_output=True, text=True).stdout
        accumulate(agg, ext, None, lambda name, k=(size, method, stage): k)
        print(f"external {jid} -> {size}/{method}/{stage}", file=sys.stderr)

    hdr = ["size", "method", "stage", "n_jobs", "n_nodes", "cpu_hours", "gpu_hours",
           "node_hours", "wall_hours", "cpu_util_pct", "max_ram_GB", "wall_start", "wall_end"]
    print(f"{'size':6s} {'method':13s} {'stage':10s} {'jobs':>5s} {'nodes':>5s} {'CPU-h':>9s} "
          f"{'GPU-h':>7s} {'wall-h':>7s} {'util%':>6s} {'maxRAM':>7s}")
    print("-" * 92)
    rows = []
    tot = {"cpuh": 0.0, "gpuh": 0.0}
    for (size, meth, stage), a in sorted(agg.items()):
        wallh = span_hours(a["start"], a["end"])
        util = 100 * a["totcpu"] / (a["cpuh"] * 3600) if a["cpuh"] > 0 else 0.0
        print(f"{size:6s} {meth:13s} {stage:10s} {a['n']:5d} {len(a['nodes']):5d} "
              f"{a['cpuh']:9.1f} {a['gpuh']:7.1f} {wallh:7.1f} {util:6.1f} {a['maxram']:6.1f}G")
        rows.append([size, meth, stage, a["n"], len(a["nodes"]), round(a["cpuh"], 2),
                     round(a["gpuh"], 2), round(a["nodeh"], 2), round(wallh, 2),
                     round(util, 1), round(a["maxram"], 1), a["start"], a["end"]])
        tot["cpuh"] += a["cpuh"]; tot["gpuh"] += a["gpuh"]
    print("-" * 92)
    print(f"{'TOTAL':20s} {'':10s} {'':5s} {'':5s} {tot['cpuh']:9.1f} {tot['gpuh']:7.1f}")

    if args.out:
        out_rows = rows
        if args.merge and os.path.exists(args.out):
            # keep old rows whose (size,method,stage) this run didn't re-measure; only
            # rows matching the current schema width (guards a stale/older CSV).
            new_keys = {(r[0], r[1], r[2]) for r in rows}
            with open(args.out, newline="") as fh:
                rd = csv.reader(fh)
                next(rd, None)  # drop old header
                kept = [r for r in rd if len(r) == len(hdr) and (r[0], r[1], r[2]) not in new_keys]
            out_rows = sorted(kept + rows, key=lambda r: (r[0], r[1], r[2]))
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(out_rows)
        print(f"\nwrote {args.out} ({len(out_rows)} rows"
              f"{', merged' if args.merge else ''})")


if __name__ == "__main__":
    main()
