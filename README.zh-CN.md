# Land-Cover Augmentation — LoRA-align

[日本語](README.md) · **简体中文** · [English](README.en.md)

一个面向航拍图像土地利用分类的图像生成与数据增强项目。包含竹林、针叶林、水田、茶园四类图像的分类流程，重点实现竹林图像的条件生成、筛选与训练样本扩充。

**Python / PyTorch / CNN / Stable Diffusion / LoRA / scikit-learn**

## 主要功能

- **图像属性标签**：根据观测尺度、颜色、亮度、阴影和清晰度，为图像生成条件标签。
- **LoRA 领域适配**：针对竹林航拍图像进行适配，生成具有不同纹理和外观的图像。
- **条件生成与筛选**：按照属性分布采样生成条件，筛选符合设定条件的生成图像。
- **分类训练**：将生成样本加入训练集，使用 CNN 完成土地利用分类。

## 生成图片展示

以下为 LoRA-align 生成的竹林航拍风格图像，展示不同的树冠纹理、明暗与场景构成。

<table>
  <tr>
    <td><img src="docs/images/generated-bamboo-01.png" width="240" alt="Generated bamboo image 1"></td>
    <td><img src="docs/images/generated-bamboo-02.png" width="240" alt="Generated bamboo image 2"></td>
    <td><img src="docs/images/generated-bamboo-03.png" width="240" alt="Generated bamboo image 3"></td>
  </tr>
  <tr>
    <td><img src="docs/images/generated-bamboo-04.png" width="240" alt="Generated bamboo image 4"></td>
    <td><img src="docs/images/generated-bamboo-05.png" width="240" alt="Generated bamboo image 5"></td>
    <td><img src="docs/images/generated-bamboo-06.png" width="240" alt="Generated bamboo image 6"></td>
  </tr>
</table>

## 代码结构

| 目录 | 内容 |
| --- | --- |
| [`classifier/`](classifier/) | CNN 分类器、数据加载与训练 |
| [`lora/`](lora/) | 属性标签、LoRA 训练封装与条件生成 |
| [`CNN_model/`](CNN_model/) | 图像增强与生成模型实验 |
| [`evaluate.py`](evaluate.py) | 分类预测结果处理 |

## 运行

使用 Python 3.10–3.12。原始数据集、模型权重及外部训练环境需另行准备。

```bash
git clone https://github.com/zephyr-tmc/land-cover-augmentation.git
cd land-cover-augmentation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python classifier/train.py --help
```

数据格式与完整使用流程见 [RUNNING.md](docs/RUNNING.md)。

## 致谢

数据来源：[国土地理院 GSI 数据集](https://gisstar.gsi.go.jp/gsi-dataset/index_ja.html)。LoRA 训练封装调用 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)。各依赖和基础模型遵循其原有许可。
