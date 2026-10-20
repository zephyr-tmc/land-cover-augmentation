import os
import json
import csv
import argparse
from typing import List, Dict, Any, Optional


def discover_result_dirs(root: str) -> List[str]:
    found = []
    for cur, dirs, files in os.walk(root):
        if "metrics.json" in files:
            found.append(cur)
    return sorted(found)


def load_metrics(dir_path: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(dir_path, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None


def extract_row(dir_path: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    name = os.path.basename(os.path.normpath(dir_path))
    macro = metrics.get("macro", {})
    roc_auc = metrics.get("roc_auc", {})
    pr_ap = metrics.get("pr_ap", {})
    row = {
        "experiment": name,
        "accuracy": to_float(metrics.get("accuracy")),
        "macro_precision": to_float(macro.get("precision")),
        "macro_recall": to_float(macro.get("recall")),
        "macro_f1": to_float(macro.get("f1")),
        "roc_auc_micro": to_float(roc_auc.get("micro")),
        "roc_auc_macro": to_float(roc_auc.get("macro")),
        "pr_ap_micro": to_float(pr_ap.get("micro")),
        "pr_ap_macro": to_float(pr_ap.get("macro")),
    }
    return row


def write_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    cols = [
        "experiment",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc_micro",
        "roc_auc_macro",
        "pr_ap_micro",
        "pr_ap_macro",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") if r.get(c, None) is not None else "" for c in cols])


def write_markdown(rows: List[Dict[str, Any]], out_md: str) -> None:
    cols = [
        "experiment",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc_micro",
        "roc_auc_macro",
        "pr_ap_micro",
        "pr_ap_macro",
    ]
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    with open(out_md, "w") as f:
        # header
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join([":-" for _ in cols]) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c, "")) if r.get(c, None) is not None else "" for c in cols) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple evaluation result directories into CSV/Markdown table")
    parser.add_argument("dirs", nargs="*", help="结果目录（包含 metrics.json）")
    parser.add_argument("--root", type=str, default=None, help="递归扫描该根目录下所有包含 metrics.json 的目录")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录，默认取 --root 或第一个目录的上一级")
    parser.add_argument("--out-csv", type=str, default="aggregate.csv")
    parser.add_argument("--out-md", type=str, default="aggregate.md")
    parser.add_argument("--sort-by", type=str, default="macro_f1", choices=[
        "accuracy","macro_precision","macro_recall","macro_f1","roc_auc_macro","pr_ap_macro","roc_auc_micro","pr_ap_micro"
    ])
    parser.add_argument("--descending", action="store_true", help="排序是否降序（默认升序，建议加此开关对分数降序）")
    args = parser.parse_args()

    result_dirs: List[str] = []
    if args.root:
        result_dirs.extend(discover_result_dirs(os.path.abspath(args.root)))
    if args.dirs:
        result_dirs.extend([os.path.abspath(d) for d in args.dirs])
    result_dirs = sorted(set(result_dirs))
    if not result_dirs:
        raise SystemExit("未发现任何包含 metrics.json 的目录")

    rows: List[Dict[str, Any]] = []
    for d in result_dirs:
        m = load_metrics(d)
        if m is None:
            continue
        rows.append(extract_row(d, m))

    key = args.sort_by
    rows.sort(key=lambda r: (r.get(key, None) is None, r.get(key, 0.0)), reverse=args.descending)

    # 输出目录
    out_dir = args.out_dir
    if out_dir is None:
        if args.root:
            out_dir = os.path.abspath(args.root)
        else:
            out_dir = os.path.abspath(os.path.join(result_dirs[0], os.pardir))
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, args.out_csv)
    out_md = os.path.join(out_dir, args.out_md)
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()







