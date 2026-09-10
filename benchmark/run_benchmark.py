#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark.py
================

统一入口，按阶段执行 Spike-in 基准测试。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCHMARK_DIR / "config.yaml"


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print("[ERROR] 配置文件格式错误", file=sys.stderr)
        sys.exit(1)
    return cfg


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> int:
    print(f"[CMD] {' '.join(map(str, cmd))}")
    if dry_run:
        return 0
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    elapsed = time.time() - start
    print(f"[INFO] 命令{'成功' if result.returncode == 0 else '失败'} (耗时 {elapsed:.2f}s)")
    return result.returncode


def check_dependencies() -> Tuple[bool, List[str]]:
    tools = ["python", "samtools", "bowtie2", "bwa", "cnvkit.py"]
    missing = [t for t in tools if shutil.which(t) is None]
    import importlib.util
    for pkg in ["pysam", "yaml", "torch", "matplotlib", "pandas", "seaborn"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(f"python-{pkg}")
    return len(missing) == 0, missing


def phase0_check(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 0: 环境检查 ===")
    ok, missing = check_dependencies()
    if not ok:
        print("[ERROR] 缺少依赖:", ", ".join(missing))
        return 1

    genome = Path(config["genome"]["reference"])
    if not genome.exists():
        print(f"[ERROR] 参考基因组不存在: {genome}")
        return 1

    if not (genome.parent / (genome.name + ".fai")).exists():
        print("[WARN] 参考基因组 .fai 索引不存在，将在需要时自动创建")

    if not Path(str(genome) + ".bwt").exists():
        print("[WARN] BWA 索引不存在，将在 Circle-Map 流程中自动创建")

    model = Path(config["detection"]["microdna_map_model"])
    if not model.exists():
        print(f"[WARN] 模型文件不存在: {model}")

    cnvkit_ref = config["genome"].get("cnvkit_reference")
    if cnvkit_ref:
        if not Path(cnvkit_ref).exists():
            print(f"[ERROR] CNVkit 参考文件不存在: {cnvkit_ref}")
            return 1

    print("[INFO] 环境检查完成")
    return 0


def phase1_generate(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 1: 生成模拟 microDNA ===")
    script = BENCHMARK_DIR / "simulate" / "generate_microdna.py"
    if not script.exists():
        return 1
    sim = config["simulation"]
    out_dir = Path(config["experiment"]["output_dir"]) / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--genome_path", str(config["genome"]["reference"]),
        "--num_sites", str(sim.get("num_sites", 1000)),
        "--min_len", str(sim.get("min_length", 200)),
        "--max_len", str(sim.get("max_length", 800)),
        "--seed", str(config["experiment"]["seed"]),
        "--output_dir", str(out_dir),
    ]

    # 检查是否启用真实模式
    use_real = sim.get("use_real_microdna", False)
    input_dir = sim.get("input_fasta_dir")
    if use_real and input_dir:
        input_dir_path = Path(input_dir)
        if input_dir_path.is_dir():
            cmd += ["--input_fasta_dir", str(input_dir_path)]
            print(f"[INFO] 使用真实 microDNA 序列目录: {input_dir_path}")
        else:
            print(f"[WARN] 真实 microDNA 目录不存在: {input_dir_path}，将使用随机模式。")

    if config["genome"].get("gap_bed"):
        cmd += ["--gap_bed", str(config["genome"]["gap_bed"])]

    return run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def phase2_circseq(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 2: 模拟 Circle-seq reads ===")
    script = BENCHMARK_DIR / "simulate" / "simulate_circseq.py"
    if not script.exists():
        return 1
    truth_dir = Path(config["experiment"]["output_dir"]) / "truth"
    out_dir = Path(config["experiment"]["output_dir"]) / "circseq"
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = config["detection"].get("threads", 8)  # 使用 detection.threads 或默认 8
    cmd = [
        sys.executable, str(script),
        "--truth_dir", str(truth_dir),
        "--output_dir", str(out_dir),
        "--copy_numbers", ",".join(map(str, config["simulation"]["copy_numbers"])),
        "--read_length", str(config["sequencing"]["read_length"]),
        "--insert_size_mean", str(config["sequencing"]["insert_size_mean"]),
        "--insert_size_std", str(config["sequencing"]["insert_size_std"]),
        "--coverage_per_copy", str(config["circseq"]["coverage_per_copy"]),
        "--junction_enrichment_factor", str(config["circseq"]["junction_enrichment"]),
        "--error_rate", str(config["sequencing"]["error_rate"]),
        "--seed", str(config["experiment"]["seed"]),
        "--threads", str(threads),
    ]
    return run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def phase3_wgs(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 3: 模拟 WGS reads ===")
    script = BENCHMARK_DIR / "simulate" / "simulate_wgs.py"
    if not script.exists():
        return 1
    truth_bed = Path(config["experiment"]["output_dir"]) / "truth" / "microdna_truth.bed"
    out_dir = Path(config["experiment"]["output_dir"]) / "wgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = config["detection"].get("threads", 8)
    cmd = [
        sys.executable, str(script),
        "--genome_path", str(config["genome"]["reference"]),
        "--truth_bed", str(truth_bed),
        "--copy_numbers", ",".join(map(str, config["simulation"]["copy_numbers"])),
        "--background_coverage", str(config["simulation"]["background_coverage"]),
        "--read_length", str(config["sequencing"]["read_length"]),
        "--insert_size_mean", str(config["sequencing"]["insert_size_mean"]),
        "--insert_size_std", str(config["sequencing"]["insert_size_std"]),
        "--seed", str(config["experiment"]["seed"]),
        "--output_dir", str(out_dir),
        "--threads", str(threads),
    ]
    if config.get("quick", False):
        cmd += ["--background_chroms", ",".join(config["genome"]["quick_chromosomes"])]
        cmd += ["--max_microdna", str(config["simulation"].get("quick_num_sites", 100))]
    else:
        cmd += ["--background_chroms", ",".join(config["genome"]["background_chroms"])]
    return run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def phase4_circle_circseq(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 4: Circle-Map on Circle-seq ===")
    script = BENCHMARK_DIR / "detect" / "run_circlemmap.py"
    if not script.exists():
        return 1
    circseq_dir = Path(config["experiment"]["output_dir"]) / "circseq"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap"
    threads = config["detection"]["circlemmap_threads"]
    for cn in config["simulation"]["copy_numbers"]:
        r1 = circseq_dir / f"cn{cn}" / "circseq_R1.fastq"
        r2 = circseq_dir / f"cn{cn}" / "circseq_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--threads", str(threads)]
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase5_microdna_wgs(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 5: MicroDNA Map on WGS ===")
    script = BENCHMARK_DIR / "detect" / "run_microdna_map.py"
    if not script.exists():
        return 1
    wgs_dir = Path(config["experiment"]["output_dir"]) / "wgs"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "microdna_map"
    threads = config["detection"]["threads"]
    model = config["detection"]["microdna_map_model"]
    limit = config["detection"]["microdna_map_limit"]
    cnvkit_ref = config["genome"].get("cnvkit_reference")
    for cn in config["simulation"]["copy_numbers"]:
        r1 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R1.fastq"
        r2 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--model_path", str(model),
               "--limit", str(limit), "--threads", str(threads)]
        if cnvkit_ref and Path(cnvkit_ref).exists():
            cmd += ["--cnvkit_reference", str(cnvkit_ref)]
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase6_circle_wgs(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 6: Circle-Map on WGS (对照) ===")
    script = BENCHMARK_DIR / "detect" / "run_circlemmap.py"
    if not script.exists():
        return 1
    wgs_dir = Path(config["experiment"]["output_dir"]) / "wgs"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap_wgs"
    threads = config["detection"]["circlemmap_threads"]
    for cn in config["simulation"]["copy_numbers"]:
        r1 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R1.fastq"
        r2 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--threads", str(threads)]
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase7_evaluate(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 7: 评估 ===")
    truth_bed = Path(config["experiment"]["output_dir"]) / "truth" / "microdna_truth.bed"
    script = BENCHMARK_DIR / "evaluate" / "metrics.py"
    if not script.exists():
        return 1
    eval_base = Path(config["experiment"]["output_dir"]) / "evaluation"
    eval_base.mkdir(parents=True, exist_ok=True)
    lod_threshold = config["evaluation"]["lod_recall_threshold"]
    overlap_threshold = config["evaluation"]["overlap_threshold"]

    tasks = [
        ("circlemmap_circseq", Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap"),
        ("microdna_map_wgs", Path(config["experiment"]["output_dir"]) / "detect" / "microdna_map"),
        ("circlemmap_wgs", Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap_wgs"),
    ]
    for name, detected_dir in tasks:
        out_dir = eval_base / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if detected_dir.exists():
            cmd = [sys.executable, str(script), "--truth_bed", str(truth_bed),
                   "--detected_dir", str(detected_dir), "--output_dir", str(out_dir),
                   "--lod_threshold", str(lod_threshold),
                   "--overlap_threshold", str(overlap_threshold)]
            if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
                return 1
    return 0


def phase8_visualize(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 8: 可视化与报告 ===")
    eval_base = Path(config["experiment"]["output_dir"]) / "evaluation"
    vis_dir = Path(config["experiment"]["output_dir"]) / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    plot_script = BENCHMARK_DIR / "visualize" / "plot_all.py"
    if not plot_script.exists():
        return 1

    lod_cmd = [
        sys.executable, str(plot_script), "--plot", "lod",
        "--microdna_metrics", str(eval_base / "microdna_map_wgs" / "summary_metrics.tsv"),
        "--circseq_metrics", str(eval_base / "circlemmap_circseq" / "summary_metrics.tsv"),
        "--circseq_wgs_metrics", str(eval_base / "circlemmap_wgs" / "summary_metrics.tsv"),
        "--output_dir", str(vis_dir),
        "--lod_threshold", str(config["evaluation"]["lod_recall_threshold"]),
    ]
    comp_cmd = [
        sys.executable, str(plot_script), "--plot", "comparison",
        "--microdna_metrics", str(eval_base / "microdna_map_wgs" / "summary_metrics.tsv"),
        "--circseq_metrics", str(eval_base / "circlemmap_circseq" / "summary_metrics.tsv"),
        "--circseq_wgs_metrics", str(eval_base / "circlemmap_wgs" / "summary_metrics.tsv"),
        "--output_dir", str(vis_dir),
    ]
    run_cmd(lod_cmd, dry_run=dry_run, cwd=PROJECT_ROOT)
    run_cmd(comp_cmd, dry_run=dry_run, cwd=PROJECT_ROOT)

    if not dry_run:
        generate_report(config, eval_base, vis_dir)
    return 0


def generate_report(config: dict, eval_base: Path, vis_dir: Path) -> None:
    report_file = Path(config["experiment"]["output_dir"]) / "final_report.md"
    with open(report_file, "w") as f:
        f.write(f"# {config['experiment']['name']} - Benchmark Report\n\n")
        f.write(f"Seed: {config['experiment']['seed']}\n")
        f.write(f"Copy numbers: {config['simulation']['copy_numbers']}\n\n")
        methods = {
            "MicroDNA Map (WGS)": eval_base / "microdna_map_wgs" / "summary_metrics.tsv",
            "Circle-Map (Circle-seq)": eval_base / "circlemmap_circseq" / "summary_metrics.tsv",
            "Circle-Map (WGS)": eval_base / "circlemmap_wgs" / "summary_metrics.tsv",
        }
        for method, file in methods.items():
            f.write(f"## {method}\n")
            if file.exists():
                f.write("| Copy Number | Recall | Precision | F1 |\n|---|---|---|---|\n")
                with open(file) as sf:
                    lines = sf.readlines()[1:]
                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 8:
                            f.write(f"| {parts[0]} | {parts[6]} | {parts[7]} | {parts[8]} |\n")
            else:
                f.write("No results.\n")
            f.write("\n")
        f.write("## Visualizations\n")
        f.write(f"See `{vis_dir}`.\n")
    print(f"[INFO] 报告已生成: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一基准测试入口")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", type=int, choices=range(0, 9), default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config["quick"] = True
        config["simulation"]["num_sites"] = config["simulation"].get("quick_num_sites", 100)
        config["genome"]["background_chroms"] = config["genome"]["quick_chromosomes"]

    output_root = Path(config["experiment"]["output_dir"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
        config["experiment"]["output_dir"] = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    phases = {
        0: phase0_check,
        1: phase1_generate,
        2: phase2_circseq,
        3: phase3_wgs,
        4: phase4_circle_circseq,
        5: phase5_microdna_wgs,
        6: phase6_circle_wgs,
        7: phase7_evaluate,
        8: phase8_visualize,
    }

    to_run = [args.phase] if args.phase is not None else list(range(9))
    total_start = time.time()
    for p in to_run:
        ret = phases[p](config, args.dry_run)
        if ret != 0:
            print(f"[ERROR] Phase {p} 失败")
            sys.exit(1)
        print(f"[INFO] Phase {p} 完成")
    if not args.dry_run:
        print(f"[INFO] 全部流程完成，总耗时 {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()