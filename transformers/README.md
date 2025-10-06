# Transformers

## Overview

Transformers are a revolutionary deep learning architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). They have become the foundation for modern natural language processing and are increasingly used in computer vision and other domains.

## Key Concepts

### 1. Self-Attention Mechanism
- Allows the model to weigh the importance of different words in a sequence
- Computes attention scores between all pairs of tokens
- Enables parallel processing unlike sequential RNNs

### 2. Multi-Head Attention
- Multiple attention mechanisms running in parallel
- Each head learns different aspects of the relationships between tokens
- Outputs are concatenated and linearly transformed

### 3. Positional Encoding
- Injects information about token positions in the sequence
- Uses sinusoidal functions of different frequencies
- Allows the model to understand word order

### 4. Feed-Forward Networks
- Applied to each position separately and identically
- Consists of two linear transformations with ReLU activation
- Adds non-linearity to the model

## Architecture Components

- **Encoder**: Processes input sequence and creates contextualized representations
- **Decoder**: Generates output sequence using encoder representations
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Helps with gradient flow

## Mathematical Foundations

The core attention mechanism is defined as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query): What we're looking for
- K (Key): What we're comparing against
- V (Value): The actual information we retrieve
- d_k: Dimension of the key vectors (for scaling)

## Applications

- Machine Translation
- Text Summarization
- Question Answering
- Text Generation (GPT models)
- Image Classification (Vision Transformers)
- Protein Structure Prediction (AlphaFold)

## Learning Objectives

By the end of this module, you will:

1. Understand the self-attention mechanism and its advantages
2. Implement a basic transformer architecture from scratch
3. Apply pre-trained transformers for various NLP tasks
4. Fine-tune transformers on custom datasets
5. Understand positional encodings and their importance

## Prerequisites

- Linear Algebra (vectors, matrices, dot products)
- Basics of Neural Networks
- Understanding of RNNs and LSTMs (helpful but not required)
- Python programming
- PyTorch or TensorFlow familiarity

## Files in This Module

- `transformers.ipynb`: Interactive Jupyter notebook with code examples
- `requirements.txt`: Python dependencies
- `transformers.tex`: Detailed mathematical derivations and theory
- `data/`: Sample datasets for practice

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need"
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
4. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition"

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Open `transformers.ipynb` in Jupyter Lab or Notebook
3. Follow along with the code examples and experiments
4. Compile `transformers.tex` for detailed mathematical explanations
