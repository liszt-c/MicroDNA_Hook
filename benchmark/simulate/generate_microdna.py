#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_microdna.py
====================

从参考基因组随机选取位置，或从已有的真实 microDNA 序列数据库
（FASTA 文件）中读取序列，生成模拟 microDNA 的 truth BED、FASTA
和 junction 信息。

支持两种模式：
  1. 随机模式（默认）：从参考基因组随机选取区域作为 microDNA body。
  2. 真实模式：指定 --input_fasta_dir 指向包含 .fa 文件的目录，
     从这些文件中读取真实 microDNA 序列作为 body，并根据 FASTA
     头部解析染色体位置。

输出：
  - microdna_truth.bed
  - microdna_sequences.fa
  - microdna_circular_templates.fa
  - junction_info.tsv

使用示例：
  # 随机模式
  python generate_microdna.py --genome_path hg19.fa --num_sites 100

  # 真实模式
  python generate_microdna.py --genome_path hg19.fa \
      --input_fasta_dir ./datasets/eccDNA \
      --num_sites 200
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pysam

# 项目路径定位
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "truth"
DEFAULT_CONFIG = BENCHMARK_DIR / "config.yaml"
DEFAULT_FLANK = 50
DEFAULT_NUM_SITES = 1000
DEFAULT_MIN_LEN = 200
DEFAULT_MAX_LEN = 800
DEFAULT_SEED = 42
N_MAX_FRACTION = 0.10


def load_yaml_config(config_path: Path) -> dict:
    """读取 YAML 配置文件（如果存在）。"""
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("[WARN] PyYAML 未安装，将使用默认参数。", file=sys.stderr)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
        except yaml.YAMLError as e:
            print(f"[WARN] config.yaml 解析失败: {e}", file=sys.stderr)
            return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated microDNA.")
    parser.add_argument("--num_sites", type=int, default=None,
                        help="Number of microDNA sites to generate (default: from config or 1000)")
    parser.add_argument("--min_len", type=int, default=None,
                        help="Minimum microDNA body length (default: 200)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Maximum microDNA body length (default: 800)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 42)")
    parser.add_argument("--genome_path", type=str, default=None,
                        help="Path to hg19.fa reference genome")
    parser.add_argument("--gap_bed", type=str, default=None,
                        help="Optional gap BED file (e.g., UCSC gap.txt converted to BED). "
                             "If not provided, N-content filtering is used.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: benchmark/results/truth)")
    parser.add_argument("--flank", type=int, default=None,
                        help="Length of upstream/downstream flanking sequence (default: 50)")
    parser.add_argument("--input_fasta_dir", type=str, default=None,
                        help="Directory containing real microDNA FASTA files. "
                             "If provided, will use these sequences as microDNA bodies.")
    return parser.parse_args()


def weighted_chromosome_choice(chroms: List[str], lengths: List[int]) -> Tuple[str, int]:
    """按染色体长度加权随机选择一条染色体。"""
    total = sum(lengths)
    r = random.uniform(0, total)
    for chrom, length in zip(chroms, lengths):
        if r < length:
            return chrom, length
        r -= length
    return chroms[-1], lengths[-1]


def n_fraction(seq: str) -> float:
    """计算序列中 N/n 的比例。"""
    if len(seq) == 0:
        return 1.0
    return (seq.count("N") + seq.count("n")) / len(seq)


def load_gap_regions(gap_bed: Optional[str]) -> Dict[str, List[Tuple[int, int]]]:
    """读取 gap BED 文件，返回 {chrom: [(start, end), ...]}。"""
    gaps: Dict[str, List[Tuple[int, int]]] = {}
    if not gap_bed:
        return gaps
    gap_path = Path(gap_bed)
    if not gap_path.exists():
        print(f"[WARN] gap BED 文件不存在: {gap_path}", file=sys.stderr)
        return gaps
    with open(gap_path, "r") as f:
        for line in f:
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
            gaps.setdefault(chrom, []).append((start, end))
    return gaps


def overlaps_gap(chrom: str, start: int, end: int, gap_dict: Dict[str, List[Tuple[int, int]]]) -> bool:
    """检查 [start, end) 是否与 gap 区域重叠。"""
    for g_start, g_end in gap_dict.get(chrom, []):
        if start < g_end and end > g_start:
            return True
    return False


def write_fasta_record(f, header: str, sequence: str, line_width: int = 80) -> None:
    """按固定行宽写入 FASTA 记录。"""
    f.write(f">{header}\n")
    for i in range(0, len(sequence), line_width):
        f.write(sequence[i:i + line_width] + "\n")


def parse_fasta_header(header: str) -> Optional[Tuple[str, int, int]]:
    """
    从 FASTA 头部解析染色体、起始和终止位置。
    支持格式：
        >chrY:601286.0-601686.0
        >chr10:100980386-100980789
    返回 (chrom, start, end) 或 None。
    """
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]
    match = re.match(r"([^:]+):(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", header)
    if not match:
        return None
    chrom = match.group(1).strip()
    start_str = match.group(2).replace(".0", "").split(".")[0]
    end_str = match.group(3).replace(".0", "").split(".")[0]
    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError:
        return None
    if start < 0 or end <= start:
        return None
    return chrom, start, end


def read_real_microdna_fasta(input_dir: Path) -> List[Tuple[str, int, int, str]]:
    """
    读取目录下所有 .fa 文件，解析每条记录。
    返回列表，每个元素为 (chrom, start, end, sequence)。
    忽略无法解析位置信息的记录。
    """
    records = []
    for fa_file in input_dir.glob("*.fa"):
        with open(fa_file, "r", encoding="utf-8") as f:
            current_header = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_header is not None:
                        seq = "".join(current_seq).upper()
                        loc = parse_fasta_header(current_header)
                        if loc is not None:
                            chrom, start, end = loc
                            records.append((chrom, start, end, seq))
                    current_header = line
                    current_seq = []
                else:
                    if current_header is None:
                        continue
                    current_seq.append(line)
            if current_header is not None:
                seq = "".join(current_seq).upper()
                loc = parse_fasta_header(current_header)
                if loc is not None:
                    chrom, start, end = loc
                    records.append((chrom, start, end, seq))
    return records


def generate_from_real_fasta(
    fasta: pysam.FastaFile,
    records: List[Tuple[str, int, int, str]],
    num_sites: int,
    flank: int,
    output_dir: Path,
    seed: int,
    gap_dict: Dict[str, List[Tuple[int, int]]],
) -> None:
    """从真实 microDNA 序列生成 truth 文件。"""
    random.seed(seed)

    if len(records) > num_sites:
        selected = random.sample(records, num_sites)
    else:
        selected = records
        print(f"[WARN] 真实 microDNA 记录数 ({len(records)}) 少于目标数量 ({num_sites})，将使用全部记录。")

    truth_bed = output_dir / "microdna_truth.bed"
    truth_fasta = output_dir / "microdna_sequences.fa"
    circular_fasta = output_dir / "microdna_circular_templates.fa"
    junction_tsv = output_dir / "junction_info.tsv"

    with open(truth_bed, "w") as bed_f, \
         open(truth_fasta, "w") as lin_fa, \
         open(circular_fasta, "w") as cir_fa, \
         open(junction_tsv, "w") as junc_f:

        junc_f.write("id\tchr\tjunction_pos\tbody_start\tbody_end\n")

        for idx, (chrom, orig_start, orig_end, body_seq) in enumerate(selected):
            # 使用实际序列长度作为 body_len
            body_len = len(body_seq)
            start = orig_start
            end = start + body_len  # 调整坐标以匹配序列长度

            if body_len <= 0:
                continue

            if gap_dict and overlaps_gap(chrom, start, end, gap_dict):
                continue

            # 提取侧翼序列
            try:
                upstream = fasta.fetch(chrom, max(0, start - flank), start).upper()
            except Exception:
                upstream = ""
            try:
                downstream = fasta.fetch(chrom, end, min(end + flank, fasta.get_reference_length(chrom))).upper()
            except Exception:
                downstream = ""

            # 补齐侧翼
            if len(upstream) < flank:
                upstream = "N" * (flank - len(upstream)) + upstream
            if len(downstream) < flank:
                downstream = downstream + "N" * (flank - len(downstream))

            linear_seq = upstream + body_seq + downstream
            circular_seq = body_seq + downstream + upstream

            micro_id = f"microdna_{idx}"

            bed_f.write(f"{chrom}\t{start}\t{end}\t{body_len}\t{micro_id}\n")
            write_fasta_record(lin_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", linear_seq)
            write_fasta_record(cir_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", circular_seq)
            junc_f.write(f"{micro_id}\t{chrom}\t{start}\t{start}\t{end}\n")

            if (idx + 1) % 100 == 0:
                print(f"[INFO] 已处理 {idx + 1}/{len(selected)} 条真实序列")


def generate_random(
    fasta: pysam.FastaFile,
    chroms: List[str],
    lengths: List[int],
    num_sites: int,
    min_len: int,
    max_len: int,
    flank: int,
    seed: int,
    gap_dict: Dict[str, List[Tuple[int, int]]],
    output_dir: Path,
) -> None:
    """从参考基因组随机生成 microDNA truth。"""
    random.seed(seed)

    truth_bed = output_dir / "microdna_truth.bed"
    truth_fasta = output_dir / "microdna_sequences.fa"
    circular_fasta = output_dir / "microdna_circular_templates.fa"
    junction_tsv = output_dir / "junction_info.tsv"

    used_sites = set()
    with open(truth_bed, "w") as bed_f, \
         open(truth_fasta, "w") as lin_fa, \
         open(circular_fasta, "w") as cir_fa, \
         open(junction_tsv, "w") as junc_f:

        junc_f.write("id\tchr\tjunction_pos\tbody_start\tbody_end\n")
        site_idx = 0
        attempts_per_site = 5000

        while site_idx < num_sites:
            success = False
            for _ in range(attempts_per_site):
                chrom, chrom_len = weighted_chromosome_choice(chroms, lengths)
                body_len = random.randint(min_len, max_len)
                max_start = chrom_len - flank - body_len
                if max_start < flank:
                    continue
                start = random.randint(flank, max_start)
                end = start + body_len
                if (chrom, start, end) in used_sites:
                    continue
                if gap_dict and overlaps_gap(chrom, start - flank, end + flank, gap_dict):
                    continue

                try:
                    upstream = fasta.fetch(chrom, start - flank, start).upper()
                    body = fasta.fetch(chrom, start, end).upper()
                    downstream = fasta.fetch(chrom, end, end + flank).upper()
                except Exception:
                    continue

                if n_fraction(upstream) > N_MAX_FRACTION or \
                   n_fraction(body) > N_MAX_FRACTION or \
                   n_fraction(downstream) > N_MAX_FRACTION:
                    continue

                used_sites.add((chrom, start, end))
                linear_seq = upstream + body + downstream
                circular_seq = body + downstream + upstream
                micro_id = f"microdna_{site_idx}"

                bed_f.write(f"{chrom}\t{start}\t{end}\t{body_len}\t{micro_id}\n")
                write_fasta_record(lin_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", linear_seq)
                write_fasta_record(cir_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", circular_seq)
                junc_f.write(f"{micro_id}\t{chrom}\t{start}\t{start}\t{end}\n")

                if (site_idx + 1) % 100 == 0:
                    print(f"[INFO] 已生成 {site_idx + 1}/{num_sites}")
                site_idx += 1
                success = True
                break

            if not success:
                print(f"[ERROR] 位点 {site_idx} 生成失败。", file=sys.stderr)
                sys.exit(1)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(DEFAULT_CONFIG)

    genome_cfg = config.get("genome", {})
    sim_cfg = config.get("simulation", {})
    experiment_cfg = config.get("experiment", {})

    genome_path = args.genome_path or genome_cfg.get("reference")
    gap_bed = args.gap_bed or genome_cfg.get("gap_bed")
    num_sites = args.num_sites if args.num_sites is not None else sim_cfg.get("num_sites", DEFAULT_NUM_SITES)
    min_len = args.min_len if args.min_len is not None else sim_cfg.get("min_length", DEFAULT_MIN_LEN)
    max_len = args.max_len if args.max_len is not None else sim_cfg.get("max_length", DEFAULT_MAX_LEN)
    seed = args.seed if args.seed is not None else experiment_cfg.get("seed", DEFAULT_SEED)
    flank = args.flank if args.flank is not None else sim_cfg.get("flank", DEFAULT_FLANK)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if "output_dir" in experiment_cfg:
            output_dir = Path(experiment_cfg["output_dir"]) / "truth"
        else:
            output_dir = DEFAULT_OUTPUT_DIR

    if not genome_path:
        print("[ERROR] 未提供参考基因组路径。", file=sys.stderr)
        sys.exit(1)
    genome_path = Path(genome_path)
    if not genome_path.exists():
        print(f"[ERROR] 参考基因组文件不存在: {genome_path}", file=sys.stderr)
        sys.exit(1)

    if min_len <= 0 or max_len < min_len:
        print("[ERROR] 长度参数无效。", file=sys.stderr)
        sys.exit(1)
    if flank < 0:
        print("[ERROR] flank 不能为负数。", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fasta = pysam.FastaFile(str(genome_path))
    except Exception as e:
        print(f"[ERROR] 无法打开参考基因组: {e}", file=sys.stderr)
        sys.exit(1)

    gap_dict = load_gap_regions(gap_bed)

    if args.input_fasta_dir:
        input_dir = Path(args.input_fasta_dir)
        if not input_dir.is_dir():
            print(f"[ERROR] 指定的真实 FASTA 目录不存在: {input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] 从真实 microDNA FASTA 读取序列: {input_dir}")
        records = read_real_microdna_fasta(input_dir)
        print(f"[INFO] 读取到 {len(records)} 条可用的真实序列")
        if len(records) == 0:
            print("[ERROR] 未读取到任何有效的 microDNA 序列，请检查 FASTA 头部格式。", file=sys.stderr)
            sys.exit(1)

        generate_from_real_fasta(
            fasta=fasta,
            records=records,
            num_sites=num_sites,
            flank=flank,
            output_dir=output_dir,
            seed=seed,
            gap_dict=gap_dict,
        )
    else:
        print(f"[INFO] 随机模式：从参考基因组选取 microDNA 位点")
        allowed_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        chroms = [c for c in fasta.references if c in allowed_chroms]
        if not chroms:
            print("[ERROR] 参考基因组中未找到主染色体。", file=sys.stderr)
            sys.exit(1)
        lengths = [fasta.get_reference_length(c) for c in chroms]

        generate_random(
            fasta=fasta,
            chroms=chroms,
            lengths=lengths,
            num_sites=num_sites,
            min_len=min_len,
            max_len=max_len,
            flank=flank,
            seed=seed,
            gap_dict=gap_dict,
            output_dir=output_dir,
        )

    fasta.close()
    print("[INFO] 模拟 microDNA 生成完成。")


if __name__ == "__main__":
    main()