#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_microdna_map.py
===================

封装 MicroDNA Map 对 WGS spike-in 数据的检测流程。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "detect" / "microdna_map"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "save" / "6.pth"
DEFAULT_LIMIT = "0.99"
DEFAULT_THREADS = 8


def check_tool(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd: list, cwd: Optional[Path] = None, check: bool = True) -> int:
    print(f"[CMD] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}")
    return result.returncode


def build_bowtie2_index(reference_fasta: Path, output_prefix: Path) -> None:
    idx_ext = [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
    if all((str(output_prefix) + ext) and Path(str(output_prefix) + ext).exists() for ext in idx_ext):
        print(f"[INFO] Bowtie2 索引已存在: {output_prefix}")
        return
    print(f"[INFO] 构建 Bowtie2 索引: {output_prefix}")
    run_cmd(["bowtie2-build", str(reference_fasta), str(output_prefix)], check=True)


def align_and_sort_reads(r1: Path, r2: Path, ref_prefix: Path, output_bam: Path, threads: int) -> None:
    print(f"[INFO] 比对 reads: {r1.name} & {r2.name}")
    cmd = (f"bowtie2 -p {threads} -x {ref_prefix} -1 {r1} -2 {r2} --no-unal | "
           f"samtools sort -@{threads} -o {output_bam}")
    ret = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"比对失败: {ret.stderr}")
    print(f"[INFO] 比对完成: {output_bam}")


def index_bam(bam_path: Path, threads: int) -> None:
    run_cmd(["samtools", "index", "-@", str(threads), str(bam_path)], check=True)


def run_cnvkit(bam_path: Path, cnvkit_reference: Path, output_dir: Path, threads: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["cnvkit.py", "batch", "-m", "wgs", "-r", str(cnvkit_reference),
             "-p", str(threads), "-d", str(output_dir), str(bam_path)], check=True)

    cnr_files = list(output_dir.glob("*.cnr"))
    if not cnr_files:
        raise FileNotFoundError("CNVkit batch 未生成 .cnr 文件")
    cnr_file = cnr_files[0]

    result_cns = output_dir / "result.cns"
    run_cmd(["cnvkit.py", "segment", str(cnr_file), "-p", str(threads),
             "-m", "cbs", "-o", str(result_cns)], check=True)

    result_call = output_dir / "result.call.cns"
    run_cmd(["cnvkit.py", "call", str(result_cns), "-o", str(result_call)], check=True)
    return result_call


def extract_cnv_regions_fasta(cnv_call_file: Path, reference_fasta: Path,
                              output_fasta_dir: Path, min_length: int = 200) -> List[Path]:
    output_fasta_dir.mkdir(parents=True, exist_ok=True)
    if not (reference_fasta.parent / (reference_fasta.name + ".fai")).exists():
        run_cmd(["samtools", "faidx", str(reference_fasta)], check=True)

    fasta_files = []
    with open(cnv_call_file, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if len(lines) < 2:
        return fasta_files

    header = lines[0].split("\t")
    try:
        chrom_idx = header.index("chromosome")
        start_idx = header.index("start")
        end_idx = header.index("end")
    except ValueError:
        chrom_idx, start_idx, end_idx = 0, 1, 2

    for i, line in enumerate(lines[1:]):
        parts = line.split("\t")
        if len(parts) <= max(chrom_idx, start_idx, end_idx):
            continue
        chrom = parts[chrom_idx]
        try:
            start, end = int(parts[start_idx]), int(parts[end_idx])
        except ValueError:
            continue
        if end - start < min_length:
            continue
        fasta_out = output_fasta_dir / f"cnv_region_{i}.fa"
        cmd = ["samtools", "faidx", str(reference_fasta), f"{chrom}:{start}-{end}"]
        with open(fasta_out, "w") as fout:
            result = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE)
            if result.returncode != 0:
                continue
        # 添加 header（samtools faidx 已包含，但以防万一）
        with open(fasta_out, "r") as fin:
            content = fin.read()
        if not content.startswith(">"):
            content = f">{chrom}:{start}-{end}\n" + content
            with open(fasta_out, "w") as fout:
                fout.write(content)
        fasta_files.append(fasta_out)
    print(f"[INFO] 提取了 {len(fasta_files)} 个 CNV 区域 FASTA")
    return fasta_files


def run_microdna_detection_batch(fasta_files: List[Path], model_path: Path,
                                 limit: str, output_dir: Path) -> Optional[Tuple[Path, List[Path]]]:
    if not fasta_files:
        return None
    run_py = PROJECT_ROOT / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"找不到 run.py: {run_py}")

    fasta_dir = output_dir / "cnv_fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    for i, fa in enumerate(fasta_files):
        shutil.copy2(fa, fasta_dir / f"region_{i}.fa")

    model_name = model_path.name
    model_in_save = PROJECT_ROOT / "save" / model_name
    if model_path.resolve() != model_in_save.resolve():
        (PROJECT_ROOT / "save").mkdir(exist_ok=True)
        shutil.copy2(model_path, model_in_save)

    cmd = [sys.executable, str(run_py), "--pattern", "long_segment",
           "--model", model_name, "--file_path", str(fasta_dir), "--limit", limit]
    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print("[ERROR] MicroDNA Map 检测失败")
        return None
    print(f"[INFO] MicroDNA Map 检测完成，耗时 {elapsed:.2f} 秒")

    bed_files = list(fasta_dir.glob("*.bed"))
    if not bed_files:
        return None
    return fasta_dir, bed_files


def merge_beds_to_standard(bed_files: List[Path], output_bed: Path) -> None:
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_bed, "w") as fout:
        for bed in bed_files:
            with open(bed, "r") as fin:
                for line in fin:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    chrom, start, end = parts[0], parts[1], parts[2]
                    fout.write(f"{chrom}\t{start}\t{end}\tmicrodna_{count}\t1.0\n")
                    count += 1
    print(f"[INFO] 合并 BED 完成，共 {count} 条记录")


def run_microdna_map(r1: Path, r2: Path, reference: Path, output_dir: Path,
                     model_path: Path = DEFAULT_MODEL_PATH, limit: str = DEFAULT_LIMIT,
                     threads: int = DEFAULT_THREADS, cnvkit_reference: Optional[Path] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "microdna_map.log"
    total_start = time.time()

    ref_prefix = output_dir / "reference_index" / reference.stem
    ref_prefix.parent.mkdir(parents=True, exist_ok=True)
    build_bowtie2_index(reference, ref_prefix)

    bam_path = output_dir / "aligned.sorted.bam"
    align_and_sort_reads(r1, r2, ref_prefix, bam_path, threads)
    index_bam(bam_path, threads)

    if cnvkit_reference is None:
        cnvkit_reference = PROJECT_ROOT / "hg19_cnvkit_filtered_ref.cnn"
        if not cnvkit_reference.exists():
            raise FileNotFoundError("未提供 CNVkit 参考文件")

    cnvkit_dir = output_dir / "cnvkit"
    result_call = run_cnvkit(bam_path, cnvkit_reference, cnvkit_dir, threads)

    fasta_dir = output_dir / "cnv_fasta"
    fasta_files = extract_cnv_regions_fasta(result_call, reference, fasta_dir)

    final_bed = output_dir / "detected_microdna.bed"
    if not fasta_files:
        final_bed.write_text("")
        return final_bed

    result = run_microdna_detection_batch(fasta_files, model_path, limit, output_dir)
    if result is None:
        final_bed.write_text("")
        return final_bed
    _, bed_files = result
    merge_beds_to_standard(bed_files, final_bed)

    elapsed = time.time() - total_start
    with open(log_file, "a") as log:
        log.write(f"Total time: {elapsed:.2f}s\n")
    print(f"[INFO] 全部流程完成，总耗时 {elapsed:.2f} 秒")
    return final_bed


def main() -> None:
    parser = argparse.ArgumentParser(description="MicroDNA Map 检测封装")
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=str, default=DEFAULT_LIMIT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cnvkit_reference", type=Path, default=None)
    args = parser.parse_args()

    for tool in ["bowtie2", "samtools", "cnvkit.py"]:
        if not check_tool(tool):
            print(f"[ERROR] 缺少工具 {tool}", file=sys.stderr)
            sys.exit(1)

    if not args.r1.exists() or not args.r2.exists() or not args.reference.exists():
        print("[ERROR] 输入文件不存在", file=sys.stderr)
        sys.exit(1)

    try:
        bed = run_microdna_map(args.r1, args.r2, args.reference, args.output_dir,
                               args.model_path, args.limit, args.threads,
                               args.cnvkit_reference)
        print(f"[INFO] 检测完成，BED: {bed}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()