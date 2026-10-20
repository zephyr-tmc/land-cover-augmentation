import os, json, argparse, numpy as np
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd

def macro_f1_argmax(y_true, y_proba):
    """不调阈值，直接 argmax → Macro-F1"""
    y_pred = np.argmax(y_proba, axis=1)
    return f1_score(y_true, y_pred, average='macro')

def macro_f1_best_threshold_same_split(y_true, y_proba):
    """
    在同一拆分上，按 PRC 为每类选择使 F1 最大的阈值（上限、偏乐观）。
    返回：macro_f1, {class_name: best_f1}, {class_name: best_thr}
    """
    n_classes = y_proba.shape[1]
    per_cls_f1 = {}
    per_cls_thr = {}
    for k in range(n_classes):
        y_bin = (y_true == k).astype(int)
        # 需要有正样本
        if y_bin.sum() == 0:
            per_cls_f1[k] = float('nan')
            per_cls_thr[k] = float('nan')
            continue
        try:
            P, R, T = precision_recall_curve(y_bin, y_proba[:, k])
            # 与阈值对齐：去掉第一个点
            f1 = 2 * P * R / (P + R + 1e-12)
            f1_valid = f1[1:]
            if len(f1_valid) == 0:
                per_cls_f1[k] = float('nan'); per_cls_thr[k] = float('nan')
                continue
            idx = np.nanargmax(f1_valid)
            per_cls_f1[k] = float(f1_valid[idx])
            per_cls_thr[k] = float(T[idx])
        except Exception:
            per_cls_f1[k] = float('nan'); per_cls_thr[k] = float('nan')
    macro_f1 = float(np.nanmean(list(per_cls_f1.values()))) if per_cls_f1 else float('nan')
    return macro_f1, per_cls_f1, per_cls_thr

def macro_f1_with_fixed_thresholds(y_true, y_proba, thr):
    """
    给定 per-class 阈值数组 thr（长度=C），做“一对多”决策：
    - 若有多类≥阈值，取分数最高那类；
    - 若都<阈值，回退 argmax。
    """
    y_pred = np.empty_like(y_true)
    over = (y_proba >= thr[None, :])
    for i in range(len(y_true)):
        if over[i].any():
            cand = np.where(over[i])[0]
            y_pred[i] = cand[np.argmax(y_proba[i, cand])]
        else:
            y_pred[i] = int(np.argmax(y_proba[i]))
    return f1_score(y_true, y_pred, average='macro')


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

        # —— 在返回前：基于同一拆分的“PRC最佳阈值上限” Macro-F1（偏乐观，用作参考） ——
    m_f1_upper, per_f1_idx, per_thr_idx = macro_f1_best_threshold_same_split(y_true, y_prob)
    # 将 per-class 字典的 key 从索引换成类名
    per_f1_name = {class_names[k]: float(v) for k, v in per_f1_idx.items()}
    per_thr_name = {class_names[k]: float(v) for k, v in per_thr_idx.items()}

    return {
        "roc_auc": {"micro": auc_micro, "macro": auc_macro, "per_class": per_class_auc},
        "pr_ap": {"micro": ap_micro, "macro": ap_macro, "per_class": per_class_ap},
        "prc_upper_bound": {  # 新增：用于参考的上限口径
            "macro_f1": m_f1_upper,
            "per_class_best_f1": per_f1_name,
            "per_class_best_threshold": per_thr_name,
        },
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
    parser.add_argument("--val-dir", type=str, default=None, help="(可选) 验证集目录：用以挑 per-class 阈值，再在 in_dir 上评估 Macro-F1")
    args = parser.parse_args()

    y_true, y_prob, class_names = load_outputs(args.in_dir)
    # 校验与概率修正
    y_true, y_prob = _validate_inputs(y_true, y_prob, class_names)
    basic = compute_basic_metrics(y_true, y_prob, class_names)
    curves = compute_roc_prc_metrics(y_true, y_prob, class_names)

    # —— Macro-F1 (argmax) ——
    macro_f1_arg = float(macro_f1_argmax(y_true, y_prob))

    # —— 可选：验证集定阈值 → 测试集（in_dir）评估 —— 
    macro_f1_fixed = None
    per_class_thr_val = None
    if args.val_dir:
        y_true_val, y_prob_val, class_names_val = load_outputs(args.val_dir)
        # 允许 val 与 test 的类名顺序不同，这里做一次对齐检查（最简：要求一致）
        assert class_names_val == class_names, "val_dir 的 class_names 与 in_dir 不一致"
        y_true_val, y_prob_val = _validate_inputs(y_true_val, y_prob_val, class_names)
        # 在 val 上挑阈值
        thr = np.zeros(len(class_names), dtype=float)
        # 复用“上限”函数里的片段：逐类 PRC → F1 最大的阈值
        for k in range(len(class_names)):
            y_bin = (y_true_val == k).astype(int)
            if y_bin.sum() == 0:
                thr[k] = 0.5
                continue
            try:
                P, R, T = precision_recall_curve(y_bin, y_prob_val[:, k])
                f1 = 2 * P * R / (P + R + 1e-12)
                f1_valid = f1[1:]
                if len(f1_valid) == 0:
                    thr[k] = 0.5
                else:
                    idx = np.nanargmax(f1_valid)
                    thr[k] = float(T[idx])
            except Exception:
                thr[k] = 0.5
        macro_f1_fixed = float(macro_f1_with_fixed_thresholds(y_true, y_prob, thr))
        per_class_thr_val = {class_names[i]: float(thr[i]) for i in range(len(class_names))}

    metrics = {
        "num_samples": int(len(y_true)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        **basic,
        **curves,
        # 新增：Macro-F1 汇总（含不同口径）
        "macro_f1": {
            "argmax": macro_f1_arg,
            "val_fixed_thresholds": macro_f1_fixed,          # 若未提供 --val-dir 则为 null
            "prc_upper_bound_same_split": curves["prc_upper_bound"]["macro_f1"],  # 上限参考
            "per_class_best_threshold_same_split": curves["prc_upper_bound"]["per_class_best_threshold"],
            "per_class_best_f1_same_split": curves["prc_upper_bound"]["per_class_best_f1"],
            "per_class_val_thresholds": per_class_thr_val,   # 若未提供 --val-dir 则为 null
        },
    }
    save_metrics(args.in_dir, metrics, filename=args.metrics_name)
    pd.DataFrame([
    {"type": "argmax", "macro_f1": macro_f1_arg},
    {"type": "prc_upper_bound", "macro_f1": curves["prc_upper_bound"]["macro_f1"]},
    {"type": "val_fixed_thresholds", "macro_f1": macro_f1_fixed}
]).to_csv(os.path.join(args.in_dir, "macro_f1.csv"), index=False)

    if not args.no_plots:
        plot_cm(np.asarray(basic["confusion_matrix"]), class_names, os.path.join(args.in_dir, args.cm_name))
        plot_roc_ovr(y_true, y_prob, class_names, os.path.join(args.in_dir, args.roc_name))
        plot_prc_ovr(y_true, y_prob, class_names, os.path.join(args.in_dir, args.prc_name))

    # 控制台输出简短报告
        # 控制台输出简短报告（补充 Macro-F1）
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro": metrics["macro"],
        "macro_f1": {
            "argmax": metrics["macro_f1"]["argmax"],
            "val_fixed_thresholds": metrics["macro_f1"]["val_fixed_thresholds"],
            "prc_upper_bound_same_split": metrics["macro_f1"]["prc_upper_bound_same_split"],
        },
        "roc_auc": {"micro": metrics["roc_auc"]["micro"], "macro": metrics["roc_auc"]["macro"]},
        "pr_ap": {"micro": metrics["pr_ap"]["micro"], "macro": metrics["pr_ap"]["macro"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
