# 🧠 小規模LLMにおける Mixture-of-Experts（MoE）の実装と検証

<div align="center">
  <img src="/assets/images/moe_structure.png" alt="MoE構造イメージ" width="600">
  <br>
  <strong>図1. MoE概略図</strong>
</div>

---

## 1. 概要（やったこと）

LLMの計算コストを大きく増やさずに表現容量を拡張する方法として **Mixture-of-Experts（MoE）** を試し、  
**GPT系小規模モデル（TinyStories）に対して「1層のみMoE化（Top-1）」** したときの効果を検証した。  
評価指標は主に **Validation PPL**、あわせて **tokens/sec** と **ピークVRAM** を記録。

---

## 2. MoEの定式化（実装に沿った最小形）

各トークンの隠れ表現を \(h\in\mathbb{R}^d\)、専門家数を \(E\)、ルータ行列を \(W_r\in\mathbb{R}^{E\times d}\) とすると、
ルーティング確率 \(p\) は

\[
g = W_r h,\qquad p = \mathrm{softmax}(g)
\]

Top-1 では最確率の専門家 \(e^\*\) を選ぶ：

\[
e^\*=\arg\max_e p_e,\qquad y=f_{e^\*}(h)
\]

容量制約はバッチ内トークン数 \(\text{tokens\_per\_batch}\) に対して

\[
\text{capacity}=\Big\lceil \text{capacity\_factor}\times \frac{\text{tokens\_per\_batch}}{E}\Big\rceil
\]

を上限とし、超過は drop する。

偏り抑制のため、専門家の平均利用率 \(\bar p_e\) を用いた補助損失（ロードバランシング）を加える：

\[
\mathcal{L}_{\text{aux}} \propto E \sum_{e=1}^{E} \bar p_e^2
\]

最終損失は

\[
\mathcal{L}=\mathcal{L}_{\text{CE}}+\lambda\,\mathcal{L}_{\text{aux}},\qquad \lambda=0.01
\]

（実装は `loss = ce + 0.01 * aux`。ルータには必要に応じて微小ノイズ `router_jitter` を加算。）

---

## 3. 実験設定

- **データ**: `roneneldan/TinyStories`（train/validation）
- **モデル**: GPT系、\(d_{\text{model}}=512\), \(n_{\text{layer}}=6\), \(n_{\text{head}}=8\), `seq_len=256`
- **MoE化**: **第3層のFFNのみ**をMoEに置換（Top-1）
- **掃き出し**: 専門家数 \(E\in\{4,8,16\}\) を比較  
  ルータ設定は **E=8固定**で `capacity_factor ∈ {1.0, 1.25, 1.5}` × `router_jitter ∈ {0.0, 0.01, 0.05}` をグリッド探索
- **学習**: `steps=10,000`, `batch_size=16`, `bf16`, AdamW, plateau系早期終了
- **評価**: Validation **PPL**（低いほど良い）、**tokens/sec**、**peak VRAM**
- **実行環境（例）**: NVIDIA GeForce **RTX 5090**（32GB）, CUDA **12.8**

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
