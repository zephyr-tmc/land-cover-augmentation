# dataset.py
import os
from collections import Counter
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, mode: str = "preprocess_only"):
    """
    mode:
      - 'preprocess_only': 仅做统一尺寸与归一化（基线0）
      - 'traditional':    传统图像增强（基线1），使用你指定的轻量策略
    """
    if mode == "preprocess_only":
        tfm = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif mode == "traditional":
        # 在线增强策略按你指定的版本：
        # Resize + CenterCrop → 轻量水平翻转 + 轻量亮度/对比度扰动
        tfm = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return tfm


class ImagePathDataset(torch.utils.data.Dataset):
    """基于 (path, label) 列表的简单数据集，用于在重采样后构造训练集。"""

    def __init__(self, samples: List[tuple], transform, class_names: List[str]):
        self.samples = samples
        self.targets = [lbl for _, lbl in samples]
        self.transform = transform
        self.classes = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_is_valid_file(only_with_subdir: Optional[str]):
    """可选构造一个 is_valid_file 回调，用于仅选择特定子目录（如 'org286' 或 'val286'）下的样本。

    ImageFolder 会递归扫描子目录；当传入此回调时，将基于路径进行二次过滤。
    """
    if not only_with_subdir:
        return None

    allowed_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    sub = f"/{only_with_subdir.strip('/')}/"

    def _predicate(path: str) -> bool:
        p = path.replace("\\", "/")
        return p.lower().endswith(allowed_exts) and (sub in p)

    return _predicate


def make_dataloaders(
    root_train: str,
    root_val: str,
    root_test: Optional[str],
    img_size: int = 286,
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "preprocess_only",
    train_only_with_subdir: Optional[str] = None,
    val_only_with_subdir: Optional[str] = None,
    test_only_with_subdir: Optional[str] = None,
    extra_train_dirs: Optional[List[str]] = None,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    # 额外参数：用于控制按类重采样的策略
    rebalance_strategy: str = "none",  # "none" | "dup_original_to_target" | "fill_with_extra_to_target" | "dup_original_add_k" | "fill_with_extra_add_k"
    target_hist: Optional[Dict[str, int]] = None,
    add_hist: Optional[Dict[str, int]] = None,
):
    """
    兼容两类目录结构：
    1) 传统 ImageFolder：root_train/root_val 直接是形如 <root>/<class>/*.png
    2) 你当前的数据：<base>/<class>/{org286,val286}/*.png
       此时将 root_train 与 root_val 都指向 <base>，并通过 *_only_with_subdir 指定仅采样哪个子目录。
    """

    train_tfm = build_transforms(img_size, mode=mode)
    eval_tfm  = build_transforms(img_size, mode="preprocess_only")   # 验证/测试不做随机增强

    ds_train_main = datasets.ImageFolder(
        root=root_train,
        transform=train_tfm,
        is_valid_file=_build_is_valid_file(train_only_with_subdir),
    )
    ds_val   = datasets.ImageFolder(
        root=root_val,
        transform=eval_tfm,
        is_valid_file=_build_is_valid_file(val_only_with_subdir),
    )
    ds_test  = (
        datasets.ImageFolder(
            root=root_test,
            transform=eval_tfm,
            is_valid_file=_build_is_valid_file(test_only_with_subdir),
        ) if root_test and os.path.exists(root_test) else None
    )

    class_names = ds_train_main.classes
    num_classes = len(class_names)

    # —— 统一 val/test 的类别映射到“训练集顺序” ——
    def _remap_to_train_mapping(ds: datasets.ImageFolder, train_class_names: list[str]):
        if ds is None:
            return
        train_class_to_idx = {c: i for i, c in enumerate(train_class_names)}
        orig_idx_to_class = ds.classes  # list[str]
        new_samples = []
        new_targets = []
        # ImageFolder.samples: List[Tuple[path, class_index]]
        for path, idx in getattr(ds, 'samples', getattr(ds, 'imgs', [])):
            cls_name = orig_idx_to_class[idx]
            if cls_name not in train_class_to_idx:
                # 该类不在训练集映射中，跳过
                continue
            new_idx = train_class_to_idx[cls_name]
            new_samples.append((path, new_idx))
            new_targets.append(new_idx)
        # 覆盖映射与样本列表
        ds.classes = train_class_names
        ds.class_to_idx = {c: i for i, c in enumerate(train_class_names)}
        ds.samples = new_samples
        ds.imgs = new_samples
        ds.targets = new_targets

    _remap_to_train_mapping(ds_val, class_names)
    if ds_test is not None:
        _remap_to_train_mapping(ds_test, class_names)

    # 如提供额外训练集目录（如 GAN/Diffusion/LoRA 生成的），准备为按类补齐做候选池
    ds_extras = []
    if extra_train_dirs:
        for p in extra_train_dirs:
            if not p:
                continue
            p = os.path.abspath(p)
            if not os.path.isdir(p):
                continue
            ds_extra = datasets.ImageFolder(root=p, transform=train_tfm)
            # 允许额外目录只包含部分类别（如仅 bamboo）。这里将其映射到“训练集顺序”。
            _remap_to_train_mapping(ds_extra, class_names)
            ds_extras.append(ds_extra)

    # ====== 构造训练集：支持三种模式 ======
    if rebalance_strategy == "none":
        # 保持原有逻辑：主训练集 + 直接拼接额外目录
        ds_list = [ds_train_main] + ds_extras
        if len(ds_list) == 1:
            ds_train = ds_train_main
            train_targets = ds_train_main.targets
        else:
            ds_train = ConcatDataset(ds_list)
            train_targets = []
            for d in ds_list:
                train_targets.extend(list(getattr(d, 'targets', [])))
    else:
        # 统一从 (path, label) 列表重建训练集，便于按照目标直方图抽样
        main_samples = list(getattr(ds_train_main, "samples", getattr(ds_train_main, "imgs", [])))
        # 按类分桶
        orig_by_class = {i: [] for i in range(num_classes)}
        for p, lbl in main_samples:
            orig_by_class[lbl].append(p)

        extra_by_class = {i: [] for i in range(num_classes)}
        for ds_extra in ds_extras:
            for p, lbl in getattr(ds_extra, "samples", getattr(ds_extra, "imgs", [])):
                extra_by_class[lbl].append(p)

        # 将目标直方图从类别名映射到 index
        target_per_idx: Dict[int, int] = {}
        if target_hist is not None:
            name_to_idx = {c: i for i, c in enumerate(class_names)}
            for cls_name, tgt in target_hist.items():
                if cls_name in name_to_idx:
                    target_per_idx[name_to_idx[cls_name]] = int(tgt)

        # 将“增量直方图”(add_hist) 从类别名映射到 index（用于固定增加 K 的实验）
        add_per_idx: Dict[int, int] = {}
        if add_hist is not None:
            name_to_idx = {c: i for i, c in enumerate(class_names)}
            for cls_name, add_n in add_hist.items():
                if cls_name in name_to_idx:
                    add_per_idx[name_to_idx[cls_name]] = int(add_n)

        final_samples: List[tuple] = []

        for cls_idx, cls_name in enumerate(class_names):
            orig_list = orig_by_class.get(cls_idx, [])
            # 始终保留所有原始样本（不 downsample）
            for p in orig_list:
                final_samples.append((p, cls_idx))

            if cls_idx not in target_per_idx:
                # 对于 “*_to_target” 策略：若未在目标直方图中指定该类，直接跳过补齐
                if rebalance_strategy in ["dup_original_to_target", "fill_with_extra_to_target"]:
                    continue

            cur_n = len(orig_list)
            # 计算需要补齐的数量 need_n
            if rebalance_strategy in ["dup_original_to_target", "fill_with_extra_to_target"]:
                target_n = target_per_idx[cls_idx]
                # 只在目标大于当前数量时进行“补齐”
                need_n = max(target_n - cur_n, 0)
            elif rebalance_strategy in ["dup_original_add_k", "fill_with_extra_add_k"]:
                need_n = int(add_per_idx.get(cls_idx, 0))
            else:
                raise ValueError(f"Unknown rebalance_strategy: {rebalance_strategy}")

            if need_n <= 0:
                continue

            if rebalance_strategy in ["dup_original_to_target", "dup_original_add_k"]:
                # B1-dup：仅从原始样本中带放回重复采样
                if len(orig_list) == 0:
                    continue
                idxs = np.random.choice(len(orig_list), size=need_n, replace=True)
                for i in idxs:
                    final_samples.append((orig_list[i], cls_idx))

            elif rebalance_strategy in ["fill_with_extra_to_target", "fill_with_extra_add_k"]:
                # B2/B3/B4：优先从离线增强/生成样本中抽取，若不足再带放回补齐
                cand = extra_by_class.get(cls_idx, [])
                if len(cand) == 0:
                    # 若额外样本为空，则退化为复制原始样本
                    if len(orig_list) == 0:
                        continue
                    idxs = np.random.choice(len(orig_list), size=need_n, replace=True)
                    for i in idxs:
                        final_samples.append((orig_list[i], cls_idx))
                else:
                    if len(cand) >= need_n:
                        idxs = np.random.choice(len(cand), size=need_n, replace=False)
                        for i in idxs:
                            final_samples.append((cand[i], cls_idx))
                    else:
                        # 额外样本不足：先全部使用，再从中带放回补齐剩余
                        for p in cand:
                            final_samples.append((p, cls_idx))
                        remain = need_n - len(cand)
                        if remain > 0:
                            idxs = np.random.choice(len(cand), size=remain, replace=True)
                            for i in idxs:
                                final_samples.append((cand[i], cls_idx))
            else:
                raise ValueError(f"Unknown rebalance_strategy: {rebalance_strategy}")

        ds_train = ImagePathDataset(final_samples, transform=train_tfm, class_names=class_names)
        train_targets = ds_train.targets

    # 计算 class weights：N / (K * n_c)
    counts = Counter(train_targets)
    N = len(train_targets)
    K = num_classes
    class_weights = torch.tensor(
        [N / (K * counts[c]) for c in range(K)],
        dtype=torch.float,
    )

    # 采样器（可选）
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        # 按类别频次的倒数为样本权重
        per_class_weight = torch.tensor(
            [N / (counts[c] + 1e-12) for c in range(K)],
            dtype=torch.float,
        )
        if isinstance(ds_train, ConcatDataset):
            all_targets = []
            for d in ds_train.datasets:
                all_targets.extend(list(getattr(d, 'targets', [])))
        else:
            all_targets = list(getattr(ds_train, 'targets', []))
        sample_weights = per_class_weight[torch.tensor(all_targets, dtype=torch.long)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    # DataLoaders
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    dl_test = (
        DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
        ) if ds_test else None
    )

    # 返回类统计供日志记录
    per_class_counts = [counts[i] for i in range(num_classes)]
    return dl_train, dl_val, dl_test, num_classes, class_names, class_weights, per_class_counts
