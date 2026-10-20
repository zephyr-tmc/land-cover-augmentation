import os, json, argparse, numpy as np
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def load_outputs(in_dir: str):
    y_prob = np.load(os.path.join(in_dir, "y_prob.npy"))
    y_true = np.load(os.path.join(in_dir, "y_true.npy"))
    with open(os.path.join(in_dir, "class_names.json"), "r") as f:
        class_names = json.load(f)
    return y_true, y_prob, class_names


def _maybe_softmax(y_prob: np.ndarray) -> np.ndarray:
    # 若不是概率（每行和非1），尝试 softmax
    row_sum = y_prob.sum(axis=1, keepdims=True)
    close = np.all(np.isfinite(row_sum)) and np.all(np.abs(row_sum - 1.0) < 1e-3)
    if close and np.all((y_prob >= -1e-6) & (y_prob <= 1 + 1e-6)):
        return np.clip(y_prob, 0.0, 1.0)
    # 数值稳定 softmax
    x = y_prob - np.max(y_prob, axis=1, keepdims=True)
    ex = np.exp(x)
    ex_sum = np.sum(ex, axis=1, keepdims=True) + 1e-12
    return ex / ex_sum


def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]):
    assert y_prob.ndim == 2, f"y_prob 必须是二维 [N, C]，当前形状 {y_prob.shape}"
    assert y_true.ndim == 1, f"y_true 必须是一维 [N]，当前形状 {y_true.shape}"
    assert y_prob.shape[0] == y_true.shape[0], "样本数不匹配"
    assert y_prob.shape[1] == len(class_names), "类别数与 class_names 不匹配"
    if not np.all(np.isfinite(y_prob)):
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=0.0, neginf=0.0)
    y_prob = _maybe_softmax(y_prob)
    return y_true.astype(int), y_prob


def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names):
    labels = list(range(len(class_names)))
    y_pred = y_prob.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        digits=6,
        zero_division=0,
    )
    # per-class metrics
    per_class = []
    for i, name in enumerate(class_names):
        if name in report:
            m = report[name]
            per_class.append({
                "class": name,
                "precision": float(m.get("precision", 0.0)),
                "recall": float(m.get("recall", 0.0)),
                "f1": float(m.get("f1-score", 0.0)),
                "support": int(m.get("support", 0)),
            })
        else:
            per_class.append({"class": name, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    accuracy = float(report.get("accuracy", 0.0))
    basic = {
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "macro": {
            "precision": float(macro.get("precision", 0.0)),
            "recall": float(macro.get("recall", 0.0)),
            "f1": float(macro.get("f1-score", 0.0)),
        },
        "weighted": {
            "precision": float(weighted.get("precision", 0.0)),
            "recall": float(weighted.get("recall", 0.0)),
            "f1": float(weighted.get("f1-score", 0.0)),
        },
        "per_class": per_class,
    }
    return basic


def compute_roc_prc_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names):
    n = len(class_names)
    labels = list(range(n))
    y_onehot = label_binarize(y_true, classes=labels)

    # micro ROC AUC（若没有任何正样本则为 NaN）
    try:
        if y_onehot.sum() > 0:
            fpr_micro, tpr_micro, _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
            auc_micro = float(auc(fpr_micro, tpr_micro))
        else:
            auc_micro = float('nan')
    except Exception:
        auc_micro = float('nan')

    per_class_auc: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        y_i = y_onehot[:, i]
        # 需存在正样本且也存在负样本
        if y_i.sum() > 0 and y_i.sum() < len(y_i):
            try:
                fpr_i, tpr_i, _ = roc_curve(y_i, y_prob[:, i])
                per_class_auc[name] = float(auc(fpr_i, tpr_i))
            except Exception:
                per_class_auc[name] = float('nan')
        else:
            per_class_auc[name] = float('nan')
    auc_macro = float(np.nanmean(list(per_class_auc.values()))) if per_class_auc else float('nan')

    # PRC AP
    try:
        ap_micro = float(average_precision_score(y_onehot.ravel(), y_prob.ravel())) if y_onehot.sum() > 0 else float('nan')
    except Exception:
        ap_micro = float('nan')

    per_class_ap: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        y_i = y_onehot[:, i]
        if y_i.sum() > 0:
            try:
                ap_i = float(average_precision_score(y_i, y_prob[:, i]))
                per_class_ap[name] = ap_i
            except Exception:
                per_class_ap[name] = float('nan')
        else:
            per_class_ap[name] = float('nan')
    ap_macro = float(np.nanmean(list(per_class_ap.values()))) if per_class_ap else float('nan')

    return {
        "roc_auc": {"micro": auc_micro, "macro": auc_macro, "per_class": per_class_auc},
        "pr_ap": {"micro": ap_micro, "macro": ap_macro, "per_class": per_class_ap},
    }


def plot_cm(cm, class_names, path):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix'); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right'); plt.yticks(ticks, class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_roc_ovr(y_true, y_prob, class_names, path):
    n = len(class_names)
    labels = list(range(n))
    y_onehot = label_binarize(y_true, classes=labels)
    fig = plt.figure(figsize=(6, 5))
    # micro
    if y_onehot.sum() > 0:
        try:
            fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
            plt.plot(fpr, tpr, label=f'micro (AUC={auc(fpr,tpr):.3f})')
        except Exception:
            pass
    # per-class
    for i in range(n):
        y_i = y_onehot[:, i]
        if y_i.sum() > 0 and y_i.sum() < len(y_i):
            try:
                fpr_i, tpr_i, _ = roc_curve(y_i, y_prob[:, i])
                plt.plot(fpr_i, tpr_i, label=f'{class_names[i]} (AUC={auc(fpr_i,tpr_i):.3f})')
            except Exception:
                continue
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.title('One-vs-Rest ROC'); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_prc_ovr(y_true, y_prob, class_names, path):
    n = len(class_names)
    labels = list(range(n))
    y_onehot = label_binarize(y_true, classes=labels)
    fig = plt.figure(figsize=(6, 5))
    # micro
    if y_onehot.sum() > 0:
        try:
            prec, rec, _ = precision_recall_curve(y_onehot.ravel(), y_prob.ravel())
            ap_micro = average_precision_score(y_onehot.ravel(), y_prob.ravel())
            plt.plot(rec, prec, label=f'micro (AP={ap_micro:.3f})')
        except Exception:
            pass
    # per-class
    for i in range(n):
        y_i = y_onehot[:, i]
        if y_i.sum() > 0:
            try:
                p_i, r_i, _ = precision_recall_curve(y_i, y_prob[:, i])
                ap_i = average_precision_score(y_i, y_prob[:, i])
                plt.plot(r_i, p_i, label=f'{class_names[i]} (AP={ap_i:.3f})')
            except Exception:
                continue
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
    plt.title('One-vs-Rest PRC'); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def save_metrics(in_dir: str, metrics: Dict[str, Any], filename: str = "metrics.json"):
    path = os.path.join(in_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification outputs (cm, ROC/PRC, per-class & macro metrics)")
    parser.add_argument("in_dir", type=str, help="目录中需包含 y_prob.npy / y_true.npy / class_names.json")
    parser.add_argument("--no-plots", action="store_true", help="不生成图像，仅导出 metrics.json")
    parser.add_argument("--cm-name", type=str, default="cm.png")
    parser.add_argument("--roc-name", type=str, default="roc.png")
    parser.add_argument("--prc-name", type=str, default="prc.png")
    parser.add_argument("--metrics-name", type=str, default="metrics.json")
    args = parser.parse_args()

    y_true, y_prob, class_names = load_outputs(args.in_dir)
    # 校验与概率修正
    y_true, y_prob = _validate_inputs(y_true, y_prob, class_names)
    basic = compute_basic_metrics(y_true, y_prob, class_names)
    curves = compute_roc_prc_metrics(y_true, y_prob, class_names)

    metrics = {
        "num_samples": int(len(y_true)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        **basic,
        **curves,
    }
    save_metrics(args.in_dir, metrics, filename=args.metrics_name)

    if not args.no_plots:
        plot_cm(np.asarray(basic["confusion_matrix"]), class_names, os.path.join(args.in_dir, args.cm_name))
        plot_roc_ovr(y_true, y_prob, class_names, os.path.join(args.in_dir, args.roc_name))
        plot_prc_ovr(y_true, y_prob, class_names, os.path.join(args.in_dir, args.prc_name))

    # 控制台输出简短报告
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro": metrics["macro"],
        "roc_auc": {"micro": metrics["roc_auc"]["micro"], "macro": metrics["roc_auc"]["macro"]},
        "pr_ap": {"micro": metrics["pr_ap"]["micro"], "macro": metrics["pr_ap"]["macro"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
