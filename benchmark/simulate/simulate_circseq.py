#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_circseq.py
===================

模拟 Circle-seq 富集实验产生的 paired-end 测序数据。
支持多线程并行生成不同拷贝数的数据。
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH_DIR = BENCHMARK_DIR / "results" / "truth"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "circseq"

DEFAULT_COPY_NUMBERS = "1,5,10,50"
DEFAULT_READ_LENGTH = 150
DEFAULT_INSERT_SIZE_MEAN = 300
DEFAULT_INSERT_SIZE_STD = 50
DEFAULT_COVERAGE_PER_COPY = 30
DEFAULT_JUNCTION_ENRICHMENT = 0.7
DEFAULT_ERROR_RATE = 0.001
DEFAULT_SEED = 42
DEFAULT_THREADS = 8
JUNCTION_WINDOW = 200


def parse_fasta(file_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_id = None
    current_seq: List[str] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                if current_id is None:
                    raise ValueError("FASTA 格式错误")
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


def parse_junction_info(tsv_path: Path) -> Dict[str, Dict[str, int]]:
    info = {}
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        if header[:1] != ["id"]:
            raise ValueError("junction_info.tsv 表头错误")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            micro_id, chrom, jpos, bstart, bend = parts
            info[micro_id] = {
                "chrom": chrom,
                "junction_pos": int(jpos),
                "body_start": int(bstart),
                "body_end": int(bend),
            }
    return info


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def circular_slice(seq: str, start: int, length: int) -> str:
    circ_len = len(seq)
    start = start % circ_len
    if start + length <= circ_len:
        return seq[start:start + length]
    return seq[start:] + seq[:length - (circ_len - start)]


def introduce_errors(seq: str, error_rate: float, rng: random.Random) -> str:
    bases = ["A", "C", "G", "T"]
    new_seq = list(seq)
    for i, base in enumerate(new_seq):
        if rng.random() < error_rate:
            choices = [b for b in bases if b.upper() != base.upper()]
            new_seq[i] = rng.choice(choices)
    return "".join(new_seq)


def write_fastq_record(f, name: str, seq: str, qual: str) -> None:
    f.write(f"@{name}\n{seq}\n+\n{qual}\n")


def generate_circular_template(linear_seq: str, body_len: int, flank: int = 50) -> str:
    linear_len = len(linear_seq)
    if (linear_len - body_len) % 2 != 0:
        raise ValueError("线性序列长度无法匹配 body_len")
    flank_calc = (linear_len - body_len) // 2
    if flank_calc <= 0:
        flank_calc = flank
    upstream = linear_seq[:flank_calc]
    body = linear_seq[flank_calc:flank_calc + body_len]
    downstream = linear_seq[flank_calc + body_len:]
    return body + downstream + upstream


def simulate_one_copy_number(
    copy_number: int,
    micro_ids: List[str],
    linear_sequences: Dict[str, str],
    junction_info: Dict[str, Dict[str, int]],
    output_dir: Path,
    read_length: int,
    insert_size_mean: int,
    insert_size_std: int,
    coverage_per_copy: int,
    junction_enrichment: float,
    error_rate: float,
    seed: int,
) -> None:
    cn_dir = output_dir / f"cn{copy_number}"
    cn_dir.mkdir(parents=True, exist_ok=True)
    r1_path = cn_dir / "circseq_R1.fastq"
    r2_path = cn_dir / "circseq_R2.fastq"

    rng = random.Random(seed + copy_number)
    print(f"[INFO] 生成拷贝数 {copy_number} 的 Circle-seq 数据...")

    with open(r1_path, "w") as r1_f, open(r2_path, "w") as r2_f:
        total_pairs = 0
        for idx, micro_id in enumerate(micro_ids):
            if micro_id not in linear_sequences or micro_id not in junction_info:
                continue
            linear_seq = linear_sequences[micro_id]
            info = junction_info[micro_id]
            body_len = info["body_end"] - info["body_start"]
            if body_len <= 0:
                continue

            try:
                circular_seq = generate_circular_template(linear_seq, body_len)
            except ValueError:
                continue

            circ_len = len(circular_seq)
            if circ_len <= read_length:
                continue

            total_coverage = copy_number * coverage_per_copy
            pairs = max(1, int(round(total_coverage * body_len / (2 * read_length))))
            junction_pos = body_len
            win_start = junction_pos - JUNCTION_WINDOW
            win_end = junction_pos + JUNCTION_WINDOW

            for pair_idx in range(pairs):
                if rng.random() < junction_enrichment:
                    s = rng.randint(win_start, win_end - 1) % circ_len
                else:
                    s = rng.randint(0, circ_len - 1)

                insert_size = int(rng.gauss(insert_size_mean, insert_size_std))
                insert_size = max(insert_size, read_length)

                r1_seq = circular_slice(circular_seq, s, read_length)
                mate_start = s + insert_size - read_length
                r2_template = circular_slice(circular_seq, mate_start, read_length)
                r2_seq = reverse_complement(r2_template)

                r1_seq = introduce_errors(r1_seq, error_rate, rng)
                r2_seq = introduce_errors(r2_seq, error_rate, rng)

                qual = "I" * read_length
                read_name = f"{micro_id}_cn{copy_number}_pair{pair_idx}"
                write_fastq_record(r1_f, read_name, r1_seq, qual)
                write_fastq_record(r2_f, read_name, r2_seq, qual)
                total_pairs += 1

            if (idx + 1) % 100 == 0:
                print(f"[INFO] 已处理 {idx + 1}/{len(micro_ids)}，累计 {total_pairs} 对")

        print(f"[INFO] 拷贝数 {copy_number} 完成，共 {total_pairs} 对")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟 Circle-seq 数据")
    parser.add_argument("--truth_dir", type=str, default=str(DEFAULT_TRUTH_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--copy_numbers", type=str, default=DEFAULT_COPY_NUMBERS)
    parser.add_argument("--read_length", type=int, default=DEFAULT_READ_LENGTH)
    parser.add_argument("--insert_size_mean", type=int, default=DEFAULT_INSERT_SIZE_MEAN)
    parser.add_argument("--insert_size_std", type=int, default=DEFAULT_INSERT_SIZE_STD)
    parser.add_argument("--coverage_per_copy", type=int, default=DEFAULT_COVERAGE_PER_COPY)
    parser.add_argument("--junction_enrichment_factor", type=float, default=DEFAULT_JUNCTION_ENRICHMENT)
    parser.add_argument("--error_rate", type=float, default=DEFAULT_ERROR_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="并行生成拷贝数的线程数（默认: 8）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth_dir = Path(args.truth_dir)
    output_dir = Path(args.output_dir)

    copy_numbers = [int(x) for x in args.copy_numbers.split(",") if x.strip()]
    if not copy_numbers:
        print("[ERROR] 未提供拷贝数", file=sys.stderr)
        sys.exit(1)

    fasta_path = truth_dir / "microdna_sequences.fa"
    junc_path = truth_dir / "junction_info.tsv"
    if not fasta_path.exists() or not junc_path.exists():
        print("[ERROR] 缺少输入文件", file=sys.stderr)
        sys.exit(1)

    linear_sequences = parse_fasta(fasta_path)
    junction_info = parse_junction_info(junc_path)

    if not linear_sequences or not junction_info:
        print("[ERROR] 输入文件为空或格式错误", file=sys.stderr)
        sys.exit(1)

    micro_ids = list(junction_info.keys())
    print(f"[INFO] 读取 {len(micro_ids)} 个 microDNA")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 并行处理不同拷贝数
    max_workers = min(args.threads, len(copy_numbers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cn in copy_numbers:
            future = executor.submit(
                simulate_one_copy_number,
                cn,
                micro_ids,
                linear_sequences,
                junction_info,
                output_dir,
                args.read_length,
                args.insert_size_mean,
                args.insert_size_std,
                args.coverage_per_copy,
                args.junction_enrichment_factor,
                args.error_rate,
                args.seed,
            )
            futures.append(future)
        # 等待全部完成
        for future in as_completed(futures):
            future.result()  # 可捕获异常

    print("[INFO] 所有拷贝数模拟完成")


if __name__ == "__main__":
    main()