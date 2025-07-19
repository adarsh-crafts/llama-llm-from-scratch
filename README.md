# Building a Llama LLM From Scratch in PyTorch

A hands-on implementation of a modern, Llama-like Large Language Model to learn transformer fundamentals from first principles.

## Table of Contents
- [Project Goal](#project-goal)
- [Key Architectural Concepts](#key-architectural-concepts)
- [Setup and Usage](#setup-and-usage)
- [Project Structure](#project-structure)
- [Detailed Documentation](#detailed-documentation)

---

## Project Goal
The objective of this repository is not to create a production-ready model, but to serve as a detailed, educational implementation of a modern transformer architecture. By building each core component from scratch in PyTorch, this project explores the mechanics behind Large Language Models and serves as a portfolio piece demonstrating a deep, first-principles understanding of the technology.

---

## Key Architectural Concepts
This implementation is based on a modern, decoder-only transformer architecture and includes several state-of-the-art optimizations:

* **Multi-Head Attention:** The core mechanism allowing the model to weigh the importance of different tokens.
* **Grouped-Query Attention (GQA):** An efficient optimization that reduces the computational and memory requirements of the attention mechanism.
* **Rotary Positional Embeddings (RoPE):** A sophisticated method for encoding the relative position of tokens.
* **RMS Pre-Normalization:** A technique used to stabilize the network during training.
* **Feed-Forward Networks:** The component that processes the contextualized embeddings from the attention block.

---

## Setup and Usage
To explore this project, you can run the Jupyter notebooks which break down each component of the model.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/adarshn656/llama-from-scratch.git](https://github.com/adarshn656/llama-from-scratch.git)
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
│   └── multi_head_attention.md   # In-depth explanation of the attention mechanism
│   └── rope_explained.md   # In-depth explanation of Rotary Positional Embedding
├── notebooks/
│   ├── 01_tokenizer.ipynb
│   └── 02_multi_head_attention.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Detailed Documentation
For a deeper, mathematical breakdown of the core components, please refer to the documents in the `docs/` folder:

* **[A Deep Dive into the Multi-Head Attention Mechanism](./docs/multi_head_attention.md)**
* *(RoPE document - coming soon!)*