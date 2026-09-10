#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_wgs.py
===============

模拟标准 WGS 测序数据，包含背景 reads 和 spike-in microDNA reads。
支持多线程并行生成不同拷贝数。
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pysam

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "wgs"
DEFAULT_TRUTH_BED = BENCHMARK_DIR / "results" / "truth" / "microdna_truth.bed"

DEFAULT_COPY_NUMBERS = "1,5,10,50"
DEFAULT_READ_LENGTH = 150
DEFAULT_INSERT_MEAN = 300
DEFAULT_INSERT_STD = 50
DEFAULT_BG_COVERAGE = 30
DEFAULT_BG_CHROMS = "chr1,chr2,chr3,chr4,chr5"
DEFAULT_SEED = 42
DEFAULT_THREADS = 8
BG_SEED_OFFSET = 100000
QUALITY_CHAR = "I"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟 WGS 数据")
    parser.add_argument("--genome_path", required=True, type=str)
    parser.add_argument("--truth_bed", type=str, default=str(DEFAULT_TRUTH_BED))
    parser.add_argument("--copy_numbers", type=str, default=DEFAULT_COPY_NUMBERS)
    parser.add_argument("--background_coverage", type=float, default=DEFAULT_BG_COVERAGE)
    parser.add_argument("--read_length", type=int, default=DEFAULT_READ_LENGTH)
    parser.add_argument("--insert_size_mean", type=int, default=DEFAULT_INSERT_MEAN)
    parser.add_argument("--insert_size_std", type=int, default=DEFAULT_INSERT_STD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--background_chroms", type=str, default=DEFAULT_BG_CHROMS)
    parser.add_argument("--background_region", type=str, default=None)
    parser.add_argument("--max_microdna", type=int, default=None)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="并行生成拷贝数的线程数（默认: 8）")
    return parser.parse_args()


def parse_copy_numbers(copy_str: str) -> List[int]:
    return [int(x) for x in copy_str.split(",") if x.strip()]


def parse_truth_bed(bed_path: Path, max_microdna: Optional[int] = None) -> List[Dict]:
    microdnas = []
    with open(bed_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            chrom, start_s, end_s, length_s, micro_id = parts[:5]
            try:
                start, end, length = int(start_s), int(end_s), int(length_s)
            except ValueError:
                continue
            microdnas.append({"chrom": chrom, "start": start, "end": end, "length": length, "id": micro_id})
    if max_microdna is not None:
        microdnas = microdnas[:max_microdna]
    return microdnas


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def circular_slice(seq: str, start: int, length: int) -> str:
    seq_len = len(seq)
    start = start % seq_len
    if start + length <= seq_len:
        return seq[start:start + length]
    return seq[start:] + seq[:length - (seq_len - start)]


def write_fastq_record(f, name: str, seq: str, qual: str) -> None:
    f.write(f"@{name}\n{seq}\n+\n{qual}\n")


def choose_chrom_weighted(chrom_lengths: Dict[str, int], rng: random.Random) -> str:
    total = sum(chrom_lengths.values())
    r = rng.uniform(0, total)
    for chrom, length in chrom_lengths.items():
        if r < length:
            return chrom
        r -= length
    return list(chrom_lengths.keys())[-1]


def generate_background_reads(
    fasta: pysam.FastaFile,
    region: Dict[str, Tuple[int, int]],
    coverage: float,
    read_length: int,
    insert_mean: int,
    insert_std: int,
    rng: random.Random,
    r1_f,
    r2_f,
) -> int:
    total_bases = sum(end - start for start, end in region.values())
    pairs = int(round(coverage * total_bases / (2 * read_length)))
    if pairs < 1:
        pairs = 1
    chrom_lengths = {c: e - s for c, (s, e) in region.items()}
    count = 0
    for _ in range(pairs):
        chrom = choose_chrom_weighted(chrom_lengths, rng)
        reg_start, reg_end = region[chrom]
        insert_size = int(rng.gauss(insert_mean, insert_std))
        insert_size = max(insert_size, read_length)
        max_start = (reg_end - reg_start) - insert_size
        if max_start < 0:
            continue
        frag_start = rng.randint(0, max_start) + reg_start
        frag_end = frag_start + insert_size
        try:
            frag_seq = fasta.fetch(chrom, frag_start, frag_end).upper()
        except Exception:
            continue
        r1_seq = frag_seq[:read_length]
        r2_template = frag_seq[-read_length:]
        r2_seq = reverse_complement(r2_template)
        name = f"bg_{chrom}_{frag_start}_{frag_end}"
        qual = QUALITY_CHAR * read_length
        write_fastq_record(r1_f, name, r1_seq, qual)
        write_fastq_record(r2_f, name, r2_seq, qual)
        count += 1
    return count


def generate_spikein_reads(
    fasta: pysam.FastaFile,
    microdnas: List[Dict],
    copy_number: int,
    read_length: int,
    insert_mean: int,
    insert_std: int,
    rng: random.Random,
    r1_f,
    r2_f,
) -> int:
    total_pairs = 0
    for micro in microdnas:
        chrom, start, end = micro["chrom"], micro["start"], micro["end"]
        body_len = end - start
        if body_len <= 0:
            continue
        try:
            body_seq = fasta.fetch(chrom, start, end).upper()
        except Exception:
            continue
        pairs = max(1, int(round(copy_number * body_len / (2 * read_length))))
        for _ in range(pairs):
            insert_size = int(rng.gauss(insert_mean, insert_std))
            insert_size = max(insert_size, read_length)
            frag_start = rng.randint(0, body_len - 1)
            frag_seq = circular_slice(body_seq, frag_start, insert_size)
            r1_seq = frag_seq[:read_length]
            r2_template = frag_seq[-read_length:]
            r2_seq = reverse_complement(r2_template)
            name = f"spikein_{micro['id']}_cn{copy_number}_{_}"
            qual = QUALITY_CHAR * read_length
            write_fastq_record(r1_f, name, r1_seq, qual)
            write_fastq_record(r2_f, name, r2_seq, qual)
            total_pairs += 1
    return total_pairs


def generate_one_copy(
    genome_path: Path,
    truth_bed_path: Path,
    copy_number: int,
    background_coverage: float,
    read_length: int,
    insert_mean: int,
    insert_std: int,
    seed: int,
    output_dir: Path,
    background_chroms: str,
    background_region: Optional[str],
    max_microdna: Optional[int],
) -> None:
    """为单个拷贝数生成 WGS FASTQ（在独立进程中执行）。"""
    # 打开参考基因组
    fasta = pysam.FastaFile(str(genome_path))
    try:
        microdnas = parse_truth_bed(truth_bed_path, max_microdna)
        if not microdnas:
            return

        region: Dict[str, Tuple[int, int]] = {}
        if background_region:
            region_str = background_region
            if ":" in region_str:
                chrom_part, coord_part = region_str.split(":", 1)
                if "-" in coord_part:
                    start_s, end_s = coord_part.split("-", 1)
                    region[chrom_part] = (int(start_s), int(end_s))
                else:
                    chrom_len = fasta.get_reference_length(chrom_part)
                    region[chrom_part] = (0, chrom_len)
            else:
                chrom_len = fasta.get_reference_length(region_str)
                region[region_str] = (0, chrom_len)
        else:
            for chrom in background_chroms.split(","):
                chrom = chrom.strip()
                if chrom:
                    chrom_len = fasta.get_reference_length(chrom)
                    region[chrom] = (0, chrom_len)

        cn_dir = output_dir / f"cn{copy_number}"
        cn_dir.mkdir(parents=True, exist_ok=True)
        r1_path = cn_dir / f"wgs_{copy_number}x_R1.fastq"
        r2_path = cn_dir / f"wgs_{copy_number}x_R2.fastq"

        bg_rng = random.Random(seed + BG_SEED_OFFSET)
        spike_rng = random.Random(seed + copy_number)

        with open(r1_path, "w") as r1_f, open(r2_path, "w") as r2_f:
            bg_pairs = generate_background_reads(
                fasta, region, background_coverage,
                read_length, insert_mean, insert_std,
                bg_rng, r1_f, r2_f
            )
            spike_pairs = generate_spikein_reads(
                fasta, microdnas, copy_number, read_length,
                insert_mean, insert_std, spike_rng, r1_f, r2_f
            )
        print(f"[INFO] 拷贝数 {copy_number} 完成：背景 {bg_pairs} 对，spike-in {spike_pairs} 对")
    finally:
        fasta.close()


def main() -> None:
    args = parse_args()
    copy_numbers = parse_copy_numbers(args.copy_numbers)
    if not copy_numbers:
        print("[ERROR] 未提供拷贝数", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genome_path = Path(args.genome_path)
    if not genome_path.exists():
        print(f"[ERROR] 参考基因组不存在: {genome_path}", file=sys.stderr)
        sys.exit(1)

    # 确保参考基因组有 faidx 索引（主进程执行）
    if not (genome_path.parent / (genome_path.name + ".fai")).exists():
        print("[INFO] 建立参考基因组 faidx 索引...")
        subprocess.run(["samtools", "faidx", str(genome_path)], check=True)

    truth_bed_path = Path(args.truth_bed)
    if not truth_bed_path.exists():
        print(f"[ERROR] truth BED 不存在: {truth_bed_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 拷贝数: {copy_numbers}")

    # 并行生成不同拷贝数
    max_workers = min(args.threads, len(copy_numbers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cn in copy_numbers:
            future = executor.submit(
                generate_one_copy,
                genome_path,
                truth_bed_path,
                cn,
                args.background_coverage,
                args.read_length,
                args.insert_size_mean,
                args.insert_size_std,
                args.seed,
                output_dir,
                args.background_chroms,
                args.background_region,
                args.max_microdna,
            )
            futures.append(future)
        for future in as_completed(futures):
            future.result()  # 可捕获异常


if __name__ == "__main__":
    main()