# 🧠 小規模LLMにおける Mixture-of-Experts（MoE）の実装と検証

<div align="center">
  <img src="/assets/images/moe_structure.png" alt="MoE構造イメージ" width="600">
  <br>
  <strong>図1. MoE概略図</strong>
</div>

---

# 1. 概要
LLMs を **計算量を大きく増やさずにモデルパラメータを増やす手法** として **Mixture-of-Experts（MoE）** を検証する。

本リポジトリでは 小規模 GPT 系モデル に対し、
**Baseline**（MoE なし） と **全ブロック MoE**（各 Transformer ブロックの FFN を MoE に置換・Top-1 ルーティング）を比較する。
評価は主に **Validation Perplexity（PPL）**、併せて **tokens/sec** と **peak VRAM** を記録する。

---

# 2. MoE とは
一般に、Attention 計算と FFN の間に **ルータ** を挿入し、FFN を複数の「**専門家（experts）**」として並列に持つ構成を指す。多様な知識をもつモデルを構築でき、推論時に実際に実行される（アクティブな）パラメータは、総パラメータの増加に比べて相対的に小さい。

## 2.1 ルーティング（gating）
各トークンの隠れ表現 ($\mathbf{h}\in\mathbb{R}^d$)、専門家数 ($E$)、ルータ行列 ($\mathbf{W}_r\in\mathbb{R}^{E\times d}$) に対し、

$$
\begin{aligned}
\mathbf{g} &= \mathbf{W}_r\mathbf{h}, \\
\mathbf{p} &= \text{softmax}(\mathbf{g}).
\end{aligned}
$$

ここで ($\mathbf{p}\in\mathbb{R}^E$) は各専門家を選ぶ確率分布である。

## 2.2 Top-1 専門家選択（Switch-style）
確率最大の専門家 ($e^{*}$) のみを通す：

$$
\begin{aligned}
e^{*} &= \text{argmax}_{e\in\{1,\dots,E\}} p_e, \\
\mathbf{y} &= f_{e^{*}}(\mathbf{h}).
\end{aligned}
$$

**備考**：Switch Transformer では出力を確率でスケーリングしない（$\mathbf{y}=f_{e^{*}}(\mathbf{h})$）。

## 2.3 容量制約（capacity）
バッチ内トークン数を ($N_{\text{tok}}$) とすると、各専門家の処理上限（capacity）は

$$
\text{capacity}
=\left\lceil
\text{capacity\_factor}\times \frac{N_{\text{tok}}}{E}
\right\rceil.
$$

上限を超えた割り当ては drop（棄却）する。

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
