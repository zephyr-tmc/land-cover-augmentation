# Land-Cover Augmentation — LoRA-align

[日本語](README.md) · **简体中文** · [English](README.en.md)

面向航拍土地利用分类的生成式数据增强研究。对竹林、针叶林、水田、茶园四类图像，探索如何生成与真实观测条件一致的训练样本，改善少数类别的学习。

**Python / PyTorch / CNN / GAN / Diffusion / LoRA / scikit-learn**

## 研究问题

增加样本数量之外，生成图像的尺度和外观是否与真实数据一致，同样会影响分类。本项目重点处理竹林图像中不同观测尺度混杂的问题。

## 方法与实现

```mermaid
flowchart LR
    A[训练图像] --> B[纹理特征与属性估计]
    B --> C[尺度、颜色、明暗等条件标签]
    C --> D[Stable Diffusion + LoRA]
    D --> E[约束采样]
    E --> F[统计筛选]
    F --> G[加入训练集]
    G --> H[CNN 分类与评价]
```

- **观测条件标签化**：从纹理特征估计观测尺度，将颜色、亮度、阴影、清晰度离散化为条件标签。
- **领域适配**：调用已有 Stable Diffusion／LoRA 训练实现，对竹林图像进行适配。
- **按分布生成**：根据属性条件分布进行采样，结合标签 dropout 与多组生成参数。
- **生成后筛选**：以真实图像统计量的分位数区间过滤生成图像。
- **分类对比**：包含 SmallCNN 系列分类器、追加样本、重采样、分类损失与 Macro-F1、AP、ROC-AUC 评价代码。

LoRA 训练封装调用 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)，不将基础模型及已有 LoRA 训练算法描述为自行开发。

## 毕业论文报告结果

![毕业论文报告的 Macro-F1 对比](docs/images/reported-macro-f1.png)

| 方法 | Macro-F1 | Macro-AP | ROC-AUC |
| --- | ---: | ---: | ---: |
| Real-only | 0.9244 | 0.9772 | 0.9961 |
| Oversampling | 0.9347 | 0.9807 | 0.9967 |
| Traditional DA | 0.9343 | 0.9803 | 0.9966 |
| LSGAN | 0.9397 | 0.9843 | 0.9976 |
| LoRA | 0.9363 | 0.9829 | 0.9971 |
| **LoRA-align** | **0.9486** | **0.9870** | **0.9977** |

来源为本人毕业论文“分类性能对比”表。除 Real-only 外，各方法设置为追加 224 张竹林图像。**以上是历史实验报告值，本次整理未重新训练。** 数据划分、来源、报告中的波动值及复现限制见[实验记录](docs/RESULTS.md)。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| [`classifier/`](classifier/) | 后期 SmallCNN 系列分类器、数据加载与训练 |
| [`lora/`](lora/) | 属性标签、LoRA 训练封装、约束生成与特征可视化 |
| [`CNN_model/`](CNN_model/) | 早期 CNN、GAN、DDPM 与图像增强实验 |
| [`evaluate.py`](evaluate.py) | 从保存的预测计算指标、绘图 |
| [`docs/`](docs/) | 使用说明、实验结果、来源及验证记录 |

早期与后期实验设置不同，早期脚本的默认值不能直接视为毕业论文的复现配置。

## 运行

预期使用 Python 3.10–3.12。仓库不包含数据集、权重和外部训练框架。

```bash
git clone https://github.com/zephyr-tmc/land-cover-augmentation.git
cd land-cover-augmentation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python classifier/train.py --help
```

数据格式、训练、生成与评价示例见 [RUNNING.md](docs/RUNNING.md)。

## 验证范围

实际执行过的检查见 [VALIDATION.md](docs/VALIDATION.md)，区分合成数据的运行检查与原研究中的精度实验。

原始数据与模型需另行获取；公开摘录不包含完整研究环境、划分清单及全部日志，因此不声称能直接复现论文表格数值。

## 来源与许可

数据来源为[国土地理院 GSI 数据集](https://gisstar.gsi.go.jp/gsi-dataset/index_ja.html)，未在此重新分发。各依赖和基础模型遵循其原有许可；自有代码尚未指定许可证。
