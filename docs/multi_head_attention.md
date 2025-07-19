# A Deep Dive into the Multi-Head Attention Mechanism

> _“Attention is all you need.”_  
The engine that powers modern Large Language Models (LLMs) like **LLaMA** is the **attention mechanism**. It's a method that enables models to dynamically focus on different parts of input sequences, understanding contextual dependencies and meaning.

This document provides a principled, step-by-step breakdown of how Multi-Head Attention works under the hood — from token embeddings to context-aware representations.

---

## 🧱 From Words to Vectors

Before any computation can happen, textual inputs must be converted into a numerical form.

- **Tokens:** Tokens are units like subwords, full words, or punctuation marks.  
  _Example:_ `"The cat sat"` → `["The", "cat", "sat"]`
  
- **Embeddings:** Each token is mapped to a high-dimensional vector capturing semantic information.  
  _Example:_  
```
The → [0.1, 0.9, 0.2, ...]
cat → [0.8, 0.1, 0.4, ...]
sat → [0.3, 0.2, 0.9, ...]
```

---

## 🎯 The Core Task: Predicting the Next Word

The goal of an autoregressive LLM is to predict the next token, given all previous ones.  
To predict `"mat"` in `"The cat sat on the ___"`, the model must understand the relationships between `"cat"`, `"sat"`, `"on"`, etc.

This is where **self-attention** comes into play — it allows the model to learn which past tokens are most relevant to the current one.

---

## 🔑 The Language of Attention: Query, Key, Value

Each token's embedding is transformed into three vectors:

- **Query (Q):** What this token is looking for (e.g., _“What should I pay attention to?”_)
- **Key (K):** What this token has to offer (e.g., _“How relevant am I?”_)
- **Value (V):** The actual information it contains

> In models like LLaMA, positional information is added via **Rotary Positional Embeddings (RoPE)** to both Q and K.  
> 📎 _See: [`rope_explained.md`](./rope_explained.md)_

---

## 🧠 Linear Projections of Q, K, V

These vectors are generated via learned linear transformations:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Where:

- $X \in \mathbb{R}^{T \times d_{\text{model}}}$ is the matrix of token embeddings
- $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ are learned weight matrices
- $Q, K, V \in \mathbb{R}^{T \times d_k}$

---

## 🔍 Step-by-Step: Self-Attention Computation

1. **Compute Attention Scores**

The attention mechanism computes similarity between tokens using dot product of the Query with Keys:

$$
\text{Scores} = QK^T
$$

Each row in this matrix represents how much one token attends to all others in the sequence.

2. **Apply Causal Mask**

To preserve autoregressive behavior (i.e., no peeking into the future), we apply a **causal mask**:

![Masked Scores](../assets/masked_scores.svg)

3. **Softmax Normalization**

We convert raw scores to attention weights using the softmax function:

$$
\text{AttentionWeights} = \text{softmax}\left(\frac{\text{MaskedScores}}{\sqrt{d_k}}\right)
$$

The division by $\sqrt{d_k}$ stabilizes gradients when $d_k$ is large.

4. **Weighted Sum of Values**

Final contextual embeddings are calculated as:

$$
\text{Output} = \text{AttentionWeights} \cdot V
$$

Each token’s output is a weighted sum of the values of all tokens.

---

## 🧪 Worked Example: 3-Token Self-Attention from Scratch

Let’s walk through a simple self-attention example using 3 tokens: `"The"`, `"cat"`, and `"sat"`. We'll use small 2D vectors for clarity.

### Input Embeddings
We define our tokens’ embeddings as:

| Token | Vector |
|-------|--------|
| `The` | [1, 0] |
| `cat` | [0, 1] |
| `sat` | [1, 1] |

Let’s assume identity matrices for the projection weights (i.e., $W_Q = W_K = W_V = I$), so:

- $Q = K = X$  

  ![V Matrix](../assets/projection_weights_V.svg)

---

### Step 1: Raw Attention Scores ($QK^T$)

Each token compares its Query with all Keys:

We compute the attention scores as:

$$
Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}, \quad
K^T = \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1
\end{bmatrix}
$$

Then,

$$
\text{Scores} = QK^T =
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

---

### Step 2: Causal Masking

To prevent future-token leakage, we mask upper-triangle entries:

$$
\text{Masked Scores} =
\begin{bmatrix}
1 & -\infty & -\infty \\
0 & 1 & -\infty \\
1 & 1 & 2 \\
\end{bmatrix}
$$

---

### Step 3: Softmax Normalization

Each row is softmaxed (ignoring masked `-∞` values):

Example:

- $\text{softmax}([1, -\infty, -\infty]) = [1, 0, 0]$
- $\text{softmax}([0, 1, -\infty]) = [0.268, 0.731, 0]$
- $\text{softmax}([1, 1, 2]) = [0.211, 0.211, 0.576]$

---

### 🔥 Visualizing the Attention Flow

Below are the heatmaps for each stage of attention:

<p align="center">
  <img src="..\assets\attention_heatmaps.png" alt="Attention Heatmaps" width="700"/>
</p>

- Left: raw dot products ($QK^T$)
- Middle: causal mask applied (future blocked)
- Right: softmax weights for each token

---

### Step 4: Weighted Sum of Values

Each output token embedding is computed by weighted sum over values:

![Final Output](../assets/final_weighted_sum_output.svg)

This produces contextualized representations that blend information from prior tokens.

---

> This hands-on example illustrates how each attention step works — from score calculation, to masking, to normalization, to final output.

---

## 🧩 Multi-Head Attention

Instead of performing attention once, the model runs multiple attention heads in parallel.

Each head has its own learnable $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$. Outputs from all heads are concatenated and projected:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

Where each head is:

$$
\text{head}_i = \text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})
$$

The final projection:

- $W_O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$  
  Reduces the concatenated dimension ($hd_v$) back to $d_{\text{model}}$


---

## 🚀 Optimizing with Grouped-Query Attention (GQA)

In standard Multi-Head Attention, every head has separate Q, K, V.

**Grouped-Query Attention** reduces this redundancy:

- Groups of Query heads share a smaller set of K and V projections.
- Let’s say there are 8 Query heads and only 2 Key/Value heads.
- This reduces memory and improves inference time.

Formally, queries are grouped like this:

$$
Q_i = XW_Q^{(i)},\quad K = XW_K,\quad V = XW_V \quad \text{(shared across groups)}
$$

---

## 📘 Summary

| Component        | Description |
|------------------|-------------|
| **Query (Q)**     | Represents what the token is searching for |
| **Key (K)**       | Represents what the token offers |
| **Value (V)**     | The actual content to be shared |
| **Attention**     | Weighted sum of values based on Q–K alignment |
| **Multi-Head**    | Multiple attention mechanisms in parallel |
| **GQA**           | Optimized multi-head attention with shared K/V |

---

## 🧾 References

- Vaswani et al., *Attention is All You Need*, 2017.
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023.

---

> _Feel free to explore the actual code implementation in [notebooks/02_attention_mechanism.ipynb](../notebooks/02_attention_mechanism.ipynb) for a practical walkthrough._
