#!/usr/bin/env python
"""2-batch 训练 smoke：验证默认 micro-batch 配置（batch=4, accum=8, 无 GC）的显存与耗时。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / "graph_llm" / "main.py"
BENCHMARK_RE = re.compile(r"^GRAPH_LLM_TRAIN_BENCHMARK (.+)$", re.MULTILINE)

# 与 main.py / args.py 默认一致，仅加 smoke 与 oom_fallback off
SMOKE_ARGV = [
    "--oom_fallback",
    "off",
    "--dataset_name",
    "Software",
    "--max_train_batches",
    "2",
    "--epochs",
    "1",
    "--max_eval_batches",
    "1",
    "--skip_bertscore",
    "--emit_train_benchmark",
    "--num_workers",
    "0",
]


def run_default_smoke(devices: str) -> dict:
    argv = [sys.executable, str(MAIN), "--devices", devices, *SMOKE_ARGV]
    print(f"Running: {' '.join(argv)}\n", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - start
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = BENCHMARK_RE.search(combined)
    payload = json.loads(match.group(1)) if match else None
    return {
        "name": "default_microbatch",
        "exit_code": proc.returncode,
        "wall_seconds_total": wall,
        "benchmark": payload,
        "oom": "CUDA out of memory" in combined or "OutOfMemoryError" in combined,
        "tail_log": combined[-4000:] if proc.returncode != 0 else "",
    }


def main():
    parser = argparse.ArgumentParser(description="graph_llm 默认训练配置 smoke")
    parser.add_argument("--devices", default="1", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "graph_llm" / "log" / "train_memory_benchmark.json"),
        type=str,
    )
    args = parser.parse_args()
    if args.dry_run:
        print("default_microbatch", SMOKE_ARGV)
        return 0

    result = run_default_smoke(args.devices)
    report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "devices": args.devices,
        "description": "默认 sdpa, batch_size=4, accumulation_steps=8, gradient_checkpointing=False",
        "results": [result],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bench = result.get("benchmark") or {}
    peak = bench.get("peak_gib") or {}
    peak_str = ", ".join(f"{k}={v:.2f}" for k, v in peak.items()) if peak else "n/a"
    md_path = out_path.with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                "# graph_llm 默认训练 smoke",
                "",
                f"- 生成时间: {report['generated_at']}",
                f"- devices: {args.devices}",
                f"- 配置: {report['description']}",
                "",
                "| 配置 | exit | OOM | train wall (s) | peak GiB |",
                "|------|------|-----|----------------|----------|",
                (
                    f"| default_microbatch | {result.get('exit_code')} | {result.get('oom')} | "
                    f"{bench.get('wall_seconds', 'n/a')} | {peak_str} |"
                ),
                "",
                "复现: `conda run -n fair python graph_llm/aux/benchmark_train_memory.py --devices 1`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}\nWrote {md_path}")
    return 0 if result.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
