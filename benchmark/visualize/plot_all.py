#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all.py
===========

整合所有基准测试可视化绘图函数，包括：
  - LOD 曲线
  - ROC/PR 曲线
  - 性能对比柱状图
  - 边界误差箱线图
  - 概率 KDE 分布

每个绘图功能封装为一个函数，可通过命令行参数选择调用。

使用示例：
  # 绘制 LOD 曲线
  python plot_all.py --plot lod \
      --microdna_metrics path/to/microdna_summary.tsv \
      --circseq_metrics path/to/circseq_summary.tsv \
      --circseq_wgs_metrics path/to/circseq_wgs_summary.tsv \
      --output_dir results/visualization

  # 绘制性能对比柱状图
  python plot_all.py --plot comparison \
      --microdna_metrics ... --circseq_metrics ... --circseq_wgs_metrics ... \
      --output_dir results/visualization

  # 绘制 ROC/PR 曲线（需要预先计算好的数据）
  python plot_all.py --plot roc_pr \
      --roc_data path/to/roc_data.tsv \
      --output_dir results/visualization

  # 绘制边界误差箱线图
  python plot_all.py --plot boundary \
      --boundary_data path/to/boundary_errors_combined.tsv \
      --output_dir results/visualization

  # 绘制概率 KDE 分布
  python plot_all.py --plot kde \
      --kde_data path/to/microdna_scores.tsv \
      --output_dir results/visualization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置全局样式
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# 配色方案
COLORS = {
    "MicroDNA Map": "#E63946",
    "Circle-Map (Circle-seq)": "#457B9D",
    "Circle-Map (WGS)": "#A8DADC",
}

# 默认 LOD 阈值
DEFAULT_LOD_THRESHOLD = 0.8


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def parse_summary_metrics(file_path: Path) -> Dict[int, Dict[str, float]]:
    """
    解析 summary_metrics.tsv，返回 {copy_number: {metric: value}}。
    包含 recall, precision, f1 等。
    """
    if not file_path.exists():
        print(f"[WARN] 文件不存在: {file_path}", file=sys.stderr)
        return {}
    df = pd.read_csv(file_path, sep="\t")
    if df.empty:
        return {}
    # 确保列名统一
    df.columns = [c.strip() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        cn = int(row["copy_number"])
        metrics = {}
        for col in df.columns:
            if col != "copy_number":
                try:
                    val = float(row[col])
                except (ValueError, TypeError):
                    val = np.nan
                metrics[col] = val
        result[cn] = metrics
    return result


def parse_lod_json(file_path: Path, threshold: float) -> Optional[int]:
    """从 lod_analysis.json 中提取 LOD（首次达到阈值的拷贝数）。"""
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    lod = data.get("lod")
    return int(lod) if lod is not None else None


def save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    """保存图形为 PDF 和 PNG（300 dpi）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 已保存: {pdf_path}")
    print(f"[INFO] 已保存: {png_path}")


# -----------------------------------------------------------------------------
# 绘图函数
# -----------------------------------------------------------------------------

def plot_lod(
    microdna_metrics: Path,
    circseq_metrics: Path,
    circseq_wgs_metrics: Path,
    output_dir: Path,
    lod_threshold: float = DEFAULT_LOD_THRESHOLD,
    lod_json_microdna: Optional[Path] = None,
    lod_json_circseq: Optional[Path] = None,
    lod_json_circseq_wgs: Optional[Path] = None,
) -> None:
    """
    绘制 LOD 曲线：x 轴拷贝数（对数刻度），y 轴 Recall。
    三条线分别对应三种方法，并标注 LOD 点。
    """
    methods = {
        "MicroDNA Map": microdna_metrics,
        "Circle-Map (Circle-seq)": circseq_metrics,
        "Circle-Map (WGS)": circseq_wgs_metrics,
    }
    data = {}
    for method, path in methods.items():
        recall_dict = {}
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            if not df.empty and "recall" in df.columns and "copy_number" in df.columns:
                recall_dict = dict(zip(df["copy_number"].astype(int), df["recall"].astype(float)))
        if recall_dict:
            data[method] = recall_dict
        else:
            print(f"[WARN] {method} 无有效 recall 数据，跳过")

    if not data:
        print("[ERROR] 没有任何方法的有效数据，无法绘制 LOD 曲线", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, recalls in data.items():
        sorted_cns = sorted(recalls.keys())
        sorted_recalls = [recalls[cn] for cn in sorted_cns]
        ax.plot(sorted_cns, sorted_recalls, marker="o", linewidth=2,
                color=COLORS.get(method, "#000000"), label=method)

    # 标注 LOD
    lod_files = {
        "MicroDNA Map": lod_json_microdna,
        "Circle-Map (Circle-seq)": lod_json_circseq,
        "Circle-Map (WGS)": lod_json_circseq_wgs,
    }
    for method, lod_file in lod_files.items():
        if lod_file is not None and method in data:
            lod = parse_lod_json(lod_file, lod_threshold)
            if lod is not None and lod in data[method]:
                recall_at_lod = data[method][lod]
                ax.scatter([lod], [recall_at_lod], s=100, facecolors='none',
                           edgecolors=COLORS.get(method, "#000000"), linewidths=2, zorder=5)
                ax.annotate(f"LOD={lod}x", xy=(lod, recall_at_lod),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, color=COLORS.get(method, "#000000"))

    ax.set_xlabel("Copy number (x)")
    ax.set_ylabel("Recall")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 5, 10, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Limit of Detection (LOD) across copy numbers")
    plt.tight_layout()
    save_figure(fig, output_dir, "lod_curve")


def plot_comparison(
    microdna_metrics: Path,
    circseq_metrics: Path,
    circseq_wgs_metrics: Path,
    output_dir: Path,
) -> None:
    """
    绘制性能对比分组柱状图：每个拷贝数下，三种方法的 Precision, Recall, F1。
    """
    methods_data = {
        "MicroDNA Map": parse_summary_metrics(microdna_metrics),
        "Circle-Map (Circle-seq)": parse_summary_metrics(circseq_metrics),
        "Circle-Map (WGS)": parse_summary_metrics(circseq_wgs_metrics),
    }

    # 获取所有拷贝数
    all_cns = sorted(set().union(*[set(d.keys()) for d in methods_data.values() if d]))
    if not all_cns:
        print("[ERROR] 没有可用数据", file=sys.stderr)
        return

    metrics = ["precision", "recall", "f1"]
    n_methods = len(methods_data)
    n_groups = len(all_cns)
    bar_width = 0.8 / n_methods

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for m_idx, (method, data_dict) in enumerate(methods_data.items()):
            values = []
            for cn in all_cns:
                if cn in data_dict and metric in data_dict[cn]:
                    values.append(data_dict[cn][metric])
                else:
                    values.append(0.0)
            x = np.arange(n_groups) + (m_idx - (n_methods-1)/2) * bar_width
            ax.bar(x, values, bar_width, label=method, color=COLORS.get(method, "#000000"))
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([f"{cn}x" for cn in all_cns])
        ax.set_ylim(0, 1.0)
        ax.set_title(metric.capitalize())
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Performance comparison across copy numbers", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_dir, "comparison_metrics")


def plot_roc_pr(
    roc_data: Path,
    output_dir: Path,
) -> None:
    """
    绘制 ROC 和 PR 曲线。
    输入数据文件应为 TSV，列包含：copy_number, threshold, fpr, tpr, precision, recall。
    每个拷贝数生成一条曲线。
    """
    if not roc_data.exists():
        print(f"[ERROR] ROC 数据文件不存在: {roc_data}", file=sys.stderr)
        return
    df = pd.read_csv(roc_data, sep="\t")
    required_cols = {"copy_number", "fpr", "tpr", "precision", "recall"}
    if not required_cols.issubset(set(df.columns)):
        print(f"[ERROR] ROC 数据文件缺少必要列: {required_cols - set(df.columns)}", file=sys.stderr)
        return

    copy_numbers = sorted(df["copy_number"].unique())
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for cn in copy_numbers:
        sub = df[df["copy_number"] == cn].sort_values("threshold")
        ax_roc.plot(sub["fpr"], sub["tpr"], marker=".", label=f"{cn}x")
        ax_pr.plot(sub["recall"], sub["precision"], marker=".", label=f"{cn}x")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(title="Copy number")
    ax_roc.grid(alpha=0.3)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(title="Copy number")
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, output_dir, "roc_pr_curves")


def plot_boundary(
    boundary_data: Path,
    output_dir: Path,
) -> None:
    """
    绘制边界误差箱线图，按拷贝数分组。
    输入数据文件 TSV，需包含列：copy_number, error。
    """
    if not boundary_data.exists():
        print(f"[ERROR] 边界误差数据文件不存在: {boundary_data}", file=sys.stderr)
        return
    df = pd.read_csv(boundary_data, sep="\t")
    if "copy_number" not in df.columns or "error" not in df.columns:
        print("[ERROR] 边界误差数据文件缺少列: copy_number 或 error", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    df.boxplot(column="error", by="copy_number", ax=ax)
    ax.set_xlabel("Copy number")
    ax.set_ylabel("Boundary error (bp)")
    ax.set_title("Boundary error distribution")
    plt.suptitle("")
    plt.tight_layout()
    save_figure(fig, output_dir, "boundary_error_boxplot")


def plot_kde(
    kde_data: Path,
    output_dir: Path,
) -> None:
    """
    绘制 MicroDNA Map 输出概率的 KDE 分布，按 TP/FP/FN 着色。
    输入数据文件 TSV，需包含列：probability, category (TP/FP/FN)。
    """
    if not kde_data.exists():
        print(f"[ERROR] KDE 数据文件不存在: {kde_data}", file=sys.stderr)
        return
    df = pd.read_csv(kde_data, sep="\t")
    if "probability" not in df.columns or "category" not in df.columns:
        print("[ERROR] KDE 数据文件缺少列: probability 或 category", file=sys.stderr)
        return

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["TP", "FP", "FN"]
    palette = {"TP": "#2a9d8f", "FP": "#e76f51", "FN": "#f4a261"}
    for cat in categories:
        sub = df[df["category"] == cat]["probability"]
        if len(sub) > 0:
            sns.kdeplot(sub, label=cat, color=palette.get(cat, "#000000"),
                        fill=True, alpha=0.3, ax=ax)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.set_title("MicroDNA Map probability distribution by classification")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, "probability_kde")


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="整合的绘图工具")
    parser.add_argument("--plot", required=True,
                        choices=["lod", "comparison", "roc_pr", "boundary", "kde"],
                        help="要绘制的图类型")
    # LOD 和 comparison 共有参数
    parser.add_argument("--microdna_metrics", type=Path, help="MicroDNA Map 的 summary_metrics.tsv")
    parser.add_argument("--circseq_metrics", type=Path, help="Circle-Map (Circle-seq) 的 summary_metrics.tsv")
    parser.add_argument("--circseq_wgs_metrics", type=Path, help="Circle-Map (WGS) 的 summary_metrics.tsv")
    # LOD 特有参数
    parser.add_argument("--lod_threshold", type=float, default=DEFAULT_LOD_THRESHOLD,
                        help=f"LOD 阈值 (默认: {DEFAULT_LOD_THRESHOLD})")
    parser.add_argument("--lod_json_microdna", type=Path, help="MicroDNA Map 的 lod_analysis.json")
    parser.add_argument("--lod_json_circseq", type=Path, help="Circle-Map (Circle-seq) 的 lod_analysis.json")
    parser.add_argument("--lod_json_circseq_wgs", type=Path, help="Circle-Map (WGS) 的 lod_analysis.json")
    # ROC/PR 参数
    parser.add_argument("--roc_data", type=Path, help="ROC/PR 数据文件（TSV）")
    # 边界误差参数
    parser.add_argument("--boundary_data", type=Path, help="边界误差数据文件（TSV，含 copy_number 和 error 列）")
    # KDE 参数
    parser.add_argument("--kde_data", type=Path, help="KDE 数据文件（TSV，含 probability 和 category 列）")
    # 输出目录
    parser.add_argument("--output_dir", required=True, type=Path, help="输出目录")

    args = parser.parse_args()

    if args.plot == "lod":
        if not all([args.microdna_metrics, args.circseq_metrics, args.circseq_wgs_metrics]):
            parser.error("--plot lod 需要提供 --microdna_metrics, --circseq_metrics, --circseq_wgs_metrics")
        plot_lod(
            args.microdna_metrics,
            args.circseq_metrics,
            args.circseq_wgs_metrics,
            args.output_dir,
            args.lod_threshold,
            args.lod_json_microdna,
            args.lod_json_circseq,
            args.lod_json_circseq_wgs,
        )
    elif args.plot == "comparison":
        if not all([args.microdna_metrics, args.circseq_metrics, args.circseq_wgs_metrics]):
            parser.error("--plot comparison 需要提供 --microdna_metrics, --circseq_metrics, --circseq_wgs_metrics")
        plot_comparison(
            args.microdna_metrics,
            args.circseq_metrics,
            args.circseq_wgs_metrics,
            args.output_dir,
        )
    elif args.plot == "roc_pr":
        if not args.roc_data:
            parser.error("--plot roc_pr 需要提供 --roc_data")
        plot_roc_pr(args.roc_data, args.output_dir)
    elif args.plot == "boundary":
        if not args.boundary_data:
            parser.error("--plot boundary 需要提供 --boundary_data")
        plot_boundary(args.boundary_data, args.output_dir)
    elif args.plot == "kde":
        if not args.kde_data:
            parser.error("--plot kde 需要提供 --kde_data")
        plot_kde(args.kde_data, args.output_dir)


if __name__ == "__main__":
    main()