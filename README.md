# LLaMA LLM From Scratch in PyTorch

A hands-on, educational implementation of a modern, LLaMA-style Large Language Model (LLM) to learn Transformer fundamentals and architecture from first principles using PyTorch.

## Table of Contents
- [Project Goal](#project-goal)
- [Key Architectural Concepts](#key-architectural-concepts)
- [Setup and Usage](#setup-and-usage)
- [Project Structure](#project-structure)
- [Detailed Documentation](#detailed-documentation)

---

## Project Goal
The objective of this repository is not to create a production-ready LLM, but to serve as a detailed, educational implementation of a modern Transformer-based architecture inspired by Meta's LLaMA model. By building each core component from scratch in PyTorch, this project explores the internal mechanics of Large Language Models and serves as a portfolio piece demonstrating a deep, first-principles understanding of foundational LLM technologies.

---

## Key Architectural Concepts
This implementation is based on a modern, decoder-only Transformer architecture and includes several state-of-the-art optimizations commonly used in LLaMA and other leading LLMs:

* **Multi-Head Attention:** The core mechanism allowing the model to weigh the importance of different tokens within a sequence.
* **Grouped-Query Attention (GQA):** An efficient optimization used in LLaMA models that reduces the computational and memory requirements of the attention mechanism.
* **Rotary Positional Embeddings (RoPE):** A sophisticated method for encoding the relative position of tokens, adopted by models like LLaMA.
* **RMS Pre-Normalization:** A technique used to stabilize the network during training in transformer-based LLMs.
* **Feed-Forward Networks:** The component that processes the contextualized embeddings from the attention block, enabling deep learning of language representations.

---

## Setup and Usage
To explore this project and learn LLaMA architecture concepts in PyTorch, you can run the Jupyter notebooks which break down each component of the model.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adarshn656/llama-from-scratch.git
    cd llama-from-scratch
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the notebooks:**
    Open the `notebooks/` directory and run the notebooks sequentially in an environment like VS Code or Jupyter Lab.

---

## Project Structure
```
.
├── docs/
│ └── multi_head_attention.md # In-depth explanation of the attention mechanism in Transformers
│ └── rope_explained.md # In-depth explanation of Rotary Positional Embedding (RoPE)
├── notebooks/
│ ├── 01_tokenizer.ipynb
│ └── 02_multi_head_attention.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Detailed Documentation
For a deeper, mathematical breakdown of the core LLaMA-inspired components, please refer to the documents in the `docs/` folder:

* **[A Deep Dive into the Multi-Head Attention Mechanism](./docs/multi_head_attention.md)**
* *(RoPE document - coming soon!)*
