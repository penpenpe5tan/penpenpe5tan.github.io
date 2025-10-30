# 🧠 小規模LLMにおける Mixture-of-Experts（MoE）の実装と検証

<div align="center">
  <img src="/assets/images/moe_structure.png" alt="MoE構造イメージ" width="600">
  <br>
  <strong>図1. MoE概略図</strong>
</div>

---

## 1. 概要

LLM を **計算量を大きく増やさずに表現容量を拡張**する手法として **Mixture-of-Experts（MoE）** を検証。  
本リポジトリでは **GPT 系の小規模モデル（TinyStories）** に対し、  
**Baseline（MoE なし）** と **全層MoE（各 Transformer ブロックの FFN を MoE に置換・Top-1 ルーティング）** を比較する。

評価は主に **Validation Perplexity（PPL）**、併せて **tokens/sec** と **peak VRAM** を記録。

---

## 2. MoE の定式化（実装準拠の最小形）

### 2.1 ルーティング確率

各トークンの隠れ表現 \( \mathbf{h}\in\mathbb{R}^d \)、専門家数 \( E \)、ルータ行列 \( \mathbf{W}_r\in\mathbb{R}^{E\times d} \) に対し、

$$
\mathbf{g}=\mathbf{W}_r\mathbf{h},\qquad
\mathbf{p}=\mathrm{softmax}(\mathbf{g})
$$

### 2.2 Top-1 専門家選択（Switch-style）

確率最大の専門家 \( e^\* \) のみを通す：

$$
e^\*=\arg\max_e\,p_e,\qquad
\mathbf{y}=f_{e^\*}(\mathbf{h})
$$

### 2.3 容量制約（Capacity）

バッチ内トークン数を \( \text{tokens\_per\_batch} \) とすると、各専門家の処理上限は

$$
\text{capacity}
=
\left\lceil
\text{capacity\_factor}\times \frac{\text{tokens\_per\_batch}}{E}
\right\rceil
$$

超過分は **drop** する。

### 2.4 ロードバランシング補助損失（実装どおり）

ルータの「重要度」 \( \mathrm{importance}_e=\mathbb{E}[p_e] \) と実割当て「負荷」
\( \mathrm{load}_e=\mathbb{E}[\mathbf{1}\{e^\*=e\}] \) を用い、

$$
\mathcal{L}_{\text{aux}}
=
E\,\sum_{e=1}^{E}
\mathrm{importance}_e\cdot \mathrm{load}_e
$$

> 実装：`aux_loss = E * (importance * load).sum()`。

### 2.5 最終損失

$$
\mathcal{L}=
\mathcal{L}_{\text{CE}}+\lambda\,\mathcal{L}_{\text{aux}},
\qquad
\lambda=0.01
$$

> コード：`loss = ce + 0.01 * aux`。必要に応じてルータに微小ノイズ `router_jitter` を加算。

---

## 3. 実装と実験条件

- **MoE 適用**：**全層MoE**（すべての Transformer ブロックで FFN→MoE 置換、Top-1）  
  ※ 全層化フラグ：`--moe_all_layers`（`--moe_layer_index` は無視）
- **モデル**：GPT 系  
  \( d_{\text{model}}=512,\ n_{\text{layer}}=6,\ n_{\text{head}}=8,\ \text{seq\_len}=256,\ d_{\text{ff}}=4d \)
- **データ**：`roneneldan/TinyStories`（`train` / `validation`）
- **スイープ**：
  - **Experts** \( E\in\{4,8,16\} \)（全層MoEで共通）
  - **Router（E=8 固定）**：  
    \( \text{capacity\_factor}\in\{1.0,1.25,1.5\} \) × \( \text{router\_jitter}\in\{0.0,0.01,0.05\} \)
- **学習**：`steps=10,000`, `batch_size=16`, `bf16(任意)`, AdamW, plateau 早期終了
- **評価**：Validation **PPL**／**tokens/sec**／**peak VRAM**
- **環境例**：NVIDIA GeForce **RTX 5090**（32GB）, CUDA **12.8**
---

## 4. 結果

### 4.1 専門家数 \(E\) の比較（抜粋）

| run                 | best PPL | 改善率 vs base |  tok/s | peak mem |
|---------------------|---------:|---------------:|-------:|--------:|
| **baseline_s10000** | **7.084**| –              | **158,731** | 4,010MB |
| **moe_e4_s10000**   | 7.023    | **+0.87%**     | 124,978 | 4,106MB |
| **moe_e8_s10000**   | **6.942**| **+2.01%**     | 104,272 | 4,216MB |
| **moe_e16_s10000**  | 6.875    | **+2.96%**     |  78,124 | 4,443MB |

**所見**: PPL は一貫して改善。速度は専門家数に応じて低下。**E=8** が精度と速度の折衷として良好。

### 4.2 ルータ設定（**E=8**固定）

<div align="center">
  <img src="runs/2025-10-29_16-35-55/router_grid.png" alt="Best PPL Heatmap (E=8)" width="600">
  <br>
  <strong>図2. Best PPL Heatmap（E=8）</strong>
</div>

対応する最良PPLの数値：

| RJ \\ CF | 1.0 | 1.25 | **1.5** |
|---------:|----:|-----:|--------:|
| **0.00** | 7.06 | 7.00 | **6.98** |
| **0.01** | 7.04 | 7.00 | **6.99** |
| **0.05** | 7.04 | 7.03 | **6.99** |

**所見**: `capacity_factor=1.5` が常に最良。`router_jitter` の影響は今回の条件では小さい（0〜0.05で差≲0.02）。

---

## 5. まとめ（今回の条件で言えること）

- **1層のみのTop-1 MoE** でも、**小規模タスクでPPLが約1–3%改善**。  
- 速度低下とメモリ微増があるため、**E=8** が実用上のバランスとして妥当。  
- ルータ設定では **capacity_factor の寄与が大きく、1.5 が安定**。`router_jitter` は本タスク規模では効果限定的。

> 推奨既定値（今回の範囲内）  
> `--moe_num_experts 8  --moe_layer_index 3  --moe_capacity_factor 1.5  --moe_router_jitter 0.0`

---

## 6. 再現メモ

- **Baseline**: `--no_moe`（aux=0）  
- **MoE有効**: `--moe_num_experts E --moe_layer_index 3`  
  ルータ調整は `--moe_capacity_factor`, `--moe_router_jitter`  
- ログ: `metrics.csv`（`step, train_loss, aux, val_loss, val_ppl, tokens_per_sec, gpu_mem_mb`）  
- ベストスナップショット: `pytorch_model.best.bin` を保存

---

## 7. 限界と今後（今回の実験の外側は述べない）

- 今回は **単層MoE・Top-1** の最小構成に限定。  
- さらなる改善検証は **複数層MoE** や **Top-2**、専門家特化などの拡張後に評価する。

---
