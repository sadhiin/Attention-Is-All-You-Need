# Transformer [Attention-Is-All-You-Need](https://arxiv.org/abs/1706.03762)


This repository contains a comprehensive implementation of the Transformer architecture as described in the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The Transformer model has revolutionized the field of Natural Language Processing (NLP) by introducing a novel attention mechanism that allows for more efficient and effective processing of sequential data.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [Self-Attention](#self-attention)
  - [Multi-Head Attention](#multi-head-attention)
  - [Positional Encoding](#positional-encoding)
  - [Layer Normalization](#layer-normalization)
  - [Transformer Encoder](#transformer-encoder)
  - [Transformer Decoder](#transformer-decoder)
- [References](#references)

## Introduction

The Transformer model is a deep learning architecture that relies entirely on self-attention mechanisms to draw global dependencies between input and output. It is designed to handle sequential data and has been widely adopted for tasks such as machine translation, text summarization, and more.

## Repository Structure

```
01-self_attention.ipynb
02-multi_head_attention.ipynb
03-posional_encoding.ipynb
04-layer_normalization.ipynb
05-transformer-encoder.ipynb
06-transformer-decoder.ipynb
decoder.py
encoder.py
README.md
requirements.txt
```

- **01-self_attention.ipynb**: Notebook explaining the self-attention mechanism.
- **02-multi_head_attention.ipynb**: Notebook detailing the multi-head attention mechanism.
- **03-posional_encoding.ipynb**: Notebook on positional encoding.
- **04-layer_normalization.ipynb**: Notebook on layer normalization.
- **05-transformer-encoder.ipynb**: Notebook implementing the Transformer encoder.
- **06-transformer-decoder.ipynb**: Notebook implementing the Transformer decoder.
- **decoder.py**: Python script for the Transformer decoder.
- **encoder.py**: Python script for the Transformer encoder.
- **README.md**: This file.
- **requirements.txt**: List of dependencies.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

You can explore the implementation of each component of the Transformer model by running the provided Jupyter notebooks. Each notebook focuses on a specific part of the model and includes detailed explanations and code.

## Components

### Self-Attention

Self-attention allows the model to weigh the importance of different words in a sentence when encoding a word. This mechanism is implemented in

[01-self_attention.ipynb](/01-self_attention.ipynb)

.

### Multi-Head Attention

Multi-head attention extends the self-attention mechanism by allowing the model to jointly attend to information from different representation subspaces. This is implemented in

[02-multi_head_attention.ipynb](/02-multi_head_attention.ipynb)

and the `MultiHeadAttention` class in

decoder.py
.

### Positional Encoding

Since the Transformer model does not inherently understand the order of words, positional encoding is used to inject information about the relative or absolute position of tokens in the sequence. This is covered in

[03-posional_encoding.ipynb](/03-posional_encoding.ipynb)

.

### Layer Normalization

Layer normalization is used to stabilize and accelerate the training of deep neural networks. This is explained in

[04-layer_normalization.ipynb](/04-layer_normalization.ipynb)

.

### Transformer Encoder

The Transformer encoder consists of a stack of identical layers, each containing a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. This is implemented in

05-transformer-encoder.ipynb

 and the `Encoder` class in

[encoder.py](/encoder.py)

.

### Transformer Decoder

The Transformer decoder is similar to the encoder but includes an additional multi-head attention mechanism to attend to the encoder's output. This is implemented in

06-transformer-decoder.ipynb

 and the `Decoder` class in

[decoder.py](/decoder.py)

.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Advances in neural information processing systems, 30.

Feel free to explore the notebooks and scripts to gain a deeper understanding of the Transformer architecture and its components.