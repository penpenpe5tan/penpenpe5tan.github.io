# 🎨 アニメ画像生成AIの軽量ファインチューニング

![サンプル画像](/assets/images/anime_sample1.png)

## 概要
アニメ特化型のDiffusionモデルを**省計算**でファインチューニングし、
画質を維持しつつ**学習時間とVRAM**を削減する手法を検証しました。

## 使用技術
- Python / PyTorch / Hugging Face Diffusers  
- LoRA / Stable Diffusion  
- CUDA（ローカル or Colab）

## 結果（例）
| 指標 | 従来 | 改良 | 変化率 |
|------|------|------|--------|
| 学習時間 | 5.2h | 3.1h | -40% |
| VRAM使用量 | 12GB | 8.6GB | -28% |
| 画質（CLIP類似度） | 0.78 | 0.77 | ±0.01 |

## 工夫した点
- 再学習層の選択（一部層のみ更新）＋LoRAで軽量化
- 少ないGPUでも学習可能なハイパラ設定
- 失敗例（手指の崩れ/色被り）を把握し、Negative Prompt/ControlNetで対処

## デモ
- 生成前後の比較画像や短い動画をここに配置

## 今後の改善
- Temporal制約を導入した動画生成への拡張（AnimateDiffなど）
- 品質評価指標の充実（FID, LPIPS 等）