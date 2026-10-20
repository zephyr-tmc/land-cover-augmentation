
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将多个训练日志 JSON（列表形式，每个元素是一个 epoch 的字典）合并为对比 CSV。
特点：
- 支持按“输入顺序”或自定义标签顺序对 5 类实验对比：
  默认标签顺序：Original-only, Baseline1_tradition, Baseline2 GAN, Diffusion, Original-only_with_online
- 可选择“最后 N 组”或“显式区间 range（如 80-100，含端点）”
- 导出：
  1) combined_comparison.csv：逐 epoch 记录，含 is_best_for_label 标记（按 val_acc 最高）
  2) summary_best.csv：每个实验的最佳指标摘要（val_acc 最大与 val_loss 最小）
用法示例：
  python logs_to_csv.py --jsons path/A.json path/B.json path/C.json path/D.json path/E.json \
    --labels "Original-only,Baseline1_tradition,Baseline2 GAN,Diffusion,Original-only_with_online" \
    --last 20 --outdir ./results
或：
  python logs_to_csv.py --jsons A.json B.json C.json D.json E.json --range 80-100
"""
import argparse, json, os, re, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

DEFAULT_LABELS = ["Original-only","Baseline1_tradition","Baseline2 GAN","Diffusion","Original-only_with_online"]

def parse_args():
    ap = argparse.ArgumentParser(description="Merge training logs (JSON) and export comparison CSVs.")
    ap.add_argument("--jsons", nargs="+", required=True, help="JSON 文件路径列表（顺序即类别顺序）")
    ap.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS),
                    help="逗号分隔的标签名，数量需与 --jsons 一致；默认使用 5 类标准名称")
    ap.add_argument("--last", type=int, default=None, help="取最后 N 组（10 或 20 等）。与 --range 互斥")
    ap.add_argument("--range", dest="range_str", type=str, default=None,
                    help="显式区间，形如 '80-100'（含端点，基于 epoch 编号）")
    ap.add_argument("--outdir", type=str, default="./csv_out", help="输出目录")
    return ap.parse_args()

def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是列表形式的 JSON。")
    # 强制按 epoch 升序排序（以防文件顺序被打乱）
    data = sorted(data, key=lambda d: d.get("epoch", 0))
    return data

def slice_records(records: List[Dict[str, Any]], last: int=None, range_str: str=None) -> List[Dict[str, Any]]:
    if range_str and last:
        raise ValueError("请二选一：--last 或 --range")
    if range_str:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", range_str)
        if not m:
            raise ValueError("--range 格式应为 'start-end'（如 80-100）")
        start, end = int(m.group(1)), int(m.group(2))
        # 只保留该区间（含端点）的 epoch
        recs = [r for r in records if int(r.get("epoch", -1)) >= start and int(r.get("epoch", -1)) <= end]
        return recs
    # 默认：取最后 N 组（若不足 N，则全取）
    if last is None:
        last = 10
    return records[-last:] if last > 0 else records

def to_rows(label: str, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in recs:
        rows.append({
            "label": label,
            "epoch": r.get("epoch"),
            "train_loss": r.get("train_loss"),
            "train_acc": r.get("train_acc"),
            "val_loss": r.get("val_loss"),
            "val_acc": r.get("val_acc"),
        })
    return rows

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    labels = [s.strip() for s in args.labels.split(",") if s.strip() != ""]
    if len(labels) != len(args.jsons):
        raise ValueError(f"labels 数量({len(labels)})与 jsons 数量({len(args.jsons)})不一致。")

    all_rows: List[Dict[str, Any]] = []
    per_label_best = []  # 汇总最佳

    for path, label in zip(args.jsons, labels):
        records = read_json(path)
        sliced = slice_records(records, last=args.last, range_str=args.range_str)
        if len(sliced) == 0:
            print(f"[WARN] {label} 在选定范围内为空，跳过。", file=sys.stderr)
            continue
        rows = to_rows(label, sliced)
        all_rows.extend(rows)

        # 计算该 label 的最佳（val_acc 最大；val_loss 最小）
        # 最佳 val_acc
        best_acc_row = max(rows, key=lambda d: (d["val_acc"] if d["val_acc"] is not None else float("-inf"), -d["val_loss"] if d["val_loss"] is not None else 0))
        # 最小 val_loss
        best_loss_row = min(rows, key=lambda d: (d["val_loss"] if d["val_loss"] is not None else float("inf"), -d["val_acc"] if d["val_acc"] is not None else 0))
        per_label_best.append({
            "label": label,
            "best_val_acc": best_acc_row["val_acc"],
            "best_val_acc_epoch": best_acc_row["epoch"],
            "best_val_loss": best_loss_row["val_loss"],
            "best_val_loss_epoch": best_loss_row["epoch"],
        })

    if not all_rows:
        print("无数据可写出。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows).sort_values(by=["label","epoch"]).reset_index(drop=True)

    # 标注每个 label 的最佳（按 val_acc）
    df["is_best_for_label"] = False
    for best in per_label_best:
        mask = (df["label"]==best["label"]) & (df["epoch"]==best["best_val_acc_epoch"])
        df.loc[mask, "is_best_for_label"] = True

    # 导出 CSV
    combined_csv = outdir / "combined_comparison.csv"
    summary_csv = outdir / "summary_best.csv"
    df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_label_best).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] 已导出：\n - {combined_csv}\n - {summary_csv}")
    print("提示：在 Excel/Sheets 中可按 label 分组筛选，或按 is_best_for_label==True 快速定位各实验最佳。")

if __name__ == "__main__":
    main()
