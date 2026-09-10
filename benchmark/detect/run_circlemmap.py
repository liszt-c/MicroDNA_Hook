#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_circlemmap.py
=================

封装 Circle-Map 检测流程。
流程：
1. BWA 比对
2. 生成 qname 和坐标排序的 BAM
3. Circle-Map ReadExtractor
4. 候选 BAM 排序并索引
5. Circle-Map Realign
6. 解析输出为标准 BED
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "detect" / "circlemmap"

CIRCLE_MAP_CMDS = ["Circle-Map", "circle", "circle_map"]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_tool(tool: str) -> Optional[str]:
    path = shutil.which(tool)
    if path is None:
        logger.error(f"未找到工具: {tool}")
    return path


def find_circle_map() -> Optional[str]:
    for cmd in CIRCLE_MAP_CMDS:
        path = shutil.which(cmd)
        if path:
            return path
    return None


def run_command(cmd: str, description: str, cwd: Optional[Path] = None) -> int:
    logger.info(f"执行: {cmd}")
    result = subprocess.run(
        f"set -o pipefail; {cmd}",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        logger.error(f"{description} 失败: {result.stderr}")
    else:
        logger.info(f"{description} 完成")
    return result.returncode


def check_bwa_index(reference: Path) -> bool:
    """检查 BWA 索引文件是否齐全。"""
    index_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    return all(Path(str(reference) + ext).exists() for ext in index_exts)


def build_bwa_index(reference: Path) -> None:
    """若 BWA 索引缺失则构建。"""
    if check_bwa_index(reference):
        logger.info(f"BWA 索引已存在: {reference}")
        return
    logger.info(f"正在创建 BWA 索引: {reference}")
    cmd = f"bwa index {reference}"
    if run_command(cmd, "创建 BWA 索引") != 0:
        raise RuntimeError("BWA 索引创建失败")


def align_reads(r1: Path, r2: Path, reference: Path, output_dir: Path, threads: int) -> tuple[Path, Path]:
    """
    使用 BWA 比对，并生成两个 BAM：
    - aligned.qname.bam: 按 reads name 排序（供 ReadExtractor 使用）
    - aligned.sorted.bam: 按坐标排序并建立索引（供 Realign 使用）
    返回 (qname_bam, sorted_bam)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_bam = output_dir / "aligned.sorted.bam"
    qname_bam = output_dir / "aligned.qname.bam"

    if not check_tool("bwa") or not check_tool("samtools"):
        raise RuntimeError("需要 bwa 和 samtools")

    build_bwa_index(reference)

    # 比对并直接排序（坐标排序）
    cmd = f"bwa mem -t {threads} {reference} {r1} {r2} | samtools sort -@{threads} -o {sorted_bam}"
    if run_command(cmd, "比对并坐标排序") != 0:
        raise RuntimeError("比对失败")

    # 生成 qname 排序的 BAM
    cmd_qname = f"samtools sort -n -@{threads} -o {qname_bam} {sorted_bam}"
    if run_command(cmd_qname, "生成 qname 排序 BAM") != 0:
        raise RuntimeError("qname 排序失败")

    # 对坐标排序 BAM 建立索引
    if run_command(f"samtools index {sorted_bam}", "建立坐标排序 BAM 索引") != 0:
        raise RuntimeError("坐标排序 BAM 索引失败")

    return qname_bam, sorted_bam


def run_circle_map_readextractor(qname_bam: Path, output_dir: Path, circle_map_path: str) -> Path:
    """运行 Circle-Map ReadExtractor 提取候选环状 reads。"""
    candidates_bam = output_dir / "circular_read_candidates.bam"
    # Circle-Map ReadExtractor 内部会拼接当前工作目录，必须使用相对路径并切换到输出目录
    qname_rel = qname_bam.name
    candidates_rel = candidates_bam.name
    cmd = f"{circle_map_path} ReadExtractor -i {qname_rel} -o {candidates_rel}"
    if run_command(cmd, "Circle-Map ReadExtractor", cwd=output_dir) != 0:
        raise RuntimeError("Circle-Map ReadExtractor 失败")
    return candidates_bam


def sort_and_index_bam(input_bam: Path, output_dir: Path, threads: int) -> Path:
    """对候选 BAM 按坐标排序并建立索引，返回排序后的 BAM 路径。"""
    sorted_bam = output_dir / (input_bam.stem + ".sorted.bam")
    cmd = f"samtools sort -@{threads} -o {sorted_bam} {input_bam}"
    if run_command(cmd, "候选 BAM 坐标排序") != 0:
        raise RuntimeError("候选 BAM 排序失败")
    if run_command(f"samtools index {sorted_bam}", "候选 BAM 建立索引") != 0:
        raise RuntimeError("候选 BAM 索引失败")
    return sorted_bam


def run_circle_map_realign(
    candidates_sorted_bam: Path,
    qname_bam: Path,
    sorted_bam: Path,
    reference: Path,
    output_dir: Path,
    circle_map_path: str,
    threads: int,
) -> Path:
    """运行 Circle-Map Realign 检测环状 DNA。"""
    raw_output = output_dir / "circlemap_raw.bed"

    # 计算相对于 output_dir 的路径，避免 Circle-Map 内部拼接错误
    rel_candidates = candidates_sorted_bam.relative_to(output_dir)
    rel_qname = qname_bam.relative_to(output_dir)
    rel_sorted = sorted_bam.relative_to(output_dir)

    # reference 转换为绝对路径后再计算相对路径
    reference_abs = reference.resolve()
    try:
        rel_reference = reference_abs.relative_to(output_dir)
    except ValueError:
        # 如果不在 output_dir 下，使用相对路径（可能包含 ..）
        rel_reference = os.path.relpath(reference_abs, output_dir)

    raw_output_rel = "circlemap_raw.bed"

    cmd = (
        f"{circle_map_path} Realign "
        f"-t {threads} "
        f"-i {rel_candidates} "
        f"-qbam {rel_qname} "
        f"-sbam {rel_sorted} "
        f"-fasta {rel_reference} "
        f"-o {raw_output_rel}"
    )
    if run_command(cmd, "Circle-Map Realign", cwd=output_dir) != 0:
        raise RuntimeError("Circle-Map Realign 失败")
    return raw_output


def parse_circle_map_output(raw_output: Path, output_dir: Path) -> Path:
    """将 Circle-Map 原始输出转换为标准 BED。"""
    standard_bed = output_dir / "detected_microdna.bed"

    if not raw_output.exists():
        # 未检测到环状 DNA，创建空文件
        logger.warning(f"Circle-Map 未生成输出文件: {raw_output}，可能未检测到环状 DNA")
        standard_bed.touch()
        return standard_bed

    count = 0
    with open(raw_output, "r") as fin, open(standard_bed, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue
            score = "1.0"
            name = f"circle_{count}"
            fout.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\n")
            count += 1
    logger.info(f"标准化 BED 写入 {standard_bed}，共 {count} 条记录")
    return standard_bed


def run_circlemmap(r1: Path, r2: Path, reference: Path, output_dir: Path, threads: int) -> Path:
    # 1. BWA 比对，生成 qname 和 sorted BAM
    qname_bam, sorted_bam = align_reads(r1, r2, reference, output_dir, threads)

    circle_map_path = find_circle_map()
    if not circle_map_path:
        raise RuntimeError("未找到 Circle-Map 可执行文件，请安装：conda install -c bioconda circle-map")

    # 2. ReadExtractor（需要切换到输出目录并使用相对路径）
    candidates_bam = run_circle_map_readextractor(qname_bam, output_dir, circle_map_path)

    # 3. 候选 BAM 排序和索引
    candidates_sorted_bam = sort_and_index_bam(candidates_bam, output_dir, threads)

    # 4. Realign（同样在输出目录下使用相对路径）
    raw_bed = run_circle_map_realign(
        candidates_sorted_bam,
        qname_bam,
        sorted_bam,
        reference,
        output_dir,
        circle_map_path,
        threads,
    )

    # 5. 解析输出
    standard_bed = parse_circle_map_output(raw_bed, output_dir)
    return standard_bed


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Circle-Map")
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if not args.r1.exists() or not args.r2.exists():
        print("[ERROR] FASTQ 文件不存在", file=sys.stderr)
        sys.exit(1)
    if not args.reference.exists():
        print("[ERROR] 参考基因组不存在", file=sys.stderr)
        sys.exit(1)

    try:
        bed = run_circlemmap(args.r1, args.r2, args.reference, args.output_dir, args.threads)
        print(f"[INFO] 检测完成，BED: {bed}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()