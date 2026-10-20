# Land-Cover Augmentation — LoRA-align

**日本語** · [简体中文](README.zh-CN.md) · [English](README.en.md)

航空写真による土地利用分類のための画像生成・データ拡張プロジェクトです。竹林・針葉樹林・水田・茶畑の分類処理を含み、竹林画像の条件付き生成、選別、訓練サンプルの拡充を実装しています。

**Python / PyTorch / CNN / Stable Diffusion / LoRA / scikit-learn**

## 主な機能

- **画像属性のタグ付け**：観測スケール、色、明度、影、鮮明さから条件トークンを作成します。
- **LoRA による領域適応**：竹林の航空写真に適応させ、異なるテクスチャや外観の画像を生成します。
- **条件付き生成と選別**：属性分布に基づいて生成条件をサンプリングし、設定した条件に沿う画像を選別します。
- **分類モデルの学習**：生成画像を訓練セットに加え、CNN で土地利用を分類します。

## 生成画像

LoRA-align で生成した竹林の航空写真風画像です。樹冠のテクスチャ、明暗、構図の異なる例を掲載しています。

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

## コード構成

| ディレクトリ | 内容 |
| --- | --- |
| [`classifier/`](classifier/) | CNN 分類器、データローダー、学習処理 |
| [`lora/`](lora/) | 属性タグ、LoRA 学習ラッパー、条件付き生成 |
| [`CNN_model/`](CNN_model/) | 画像拡張と生成モデルの実験 |
| [`evaluate.py`](evaluate.py) | 分類モデルの予測結果の処理 |

## 実行

Python 3.10–3.12 を使用します。元データセット、モデル重み、外部の学習環境は別途用意してください。

```bash
git clone https://github.com/zephyr-tmc/land-cover-augmentation.git
cd land-cover-augmentation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python classifier/train.py --help
```

データ形式と利用手順は [RUNNING.md](docs/RUNNING.md) を参照してください。

## 謝辞

データの出典：[国土地理院 GSI データセット](https://gisstar.gsi.go.jp/gsi-dataset/index_ja.html)。LoRA 学習ラッパーは [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) を使用します。依存ライブラリと基盤モデルには、それぞれのライセンスが適用されます。
