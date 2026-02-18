# MicroGPT

**The most atomic way to train and inference a GPT in pure Python.**

A minimal, dependency-free implementation of GPT training and inference in 200 lines of Python. Everything you need to understand how large language models work, with zero abstractions hiding the fundamentals.

## What's Inside

| Component | Lines | Purpose |
|-----------|-------|---------|
| `Value` autograd | ~45 | Custom automatic differentiation from scratch |
| GPT architecture | ~60 | Multi-head attention, RMSNorm, residual connections |
| Training loop | ~30 | Adam optimizer with cosine LR decay |
| Inference | ~12 | Sampling with temperature control |

## Architecture

Based on GPT-2 with simplifications:
- **Transformer**: Multi-head self-attention, residual connections
- **Normalization**: RMSNorm (simpler than LayerNorm)
- **Activation**: ReLU² (simpler than GeLU)
- **No biases**: Fewer parameters to track
- **Custom autograd**: `Value` class implements chain rule for backprop

## Hyperparameters

```python
n_embd = 16      # embedding dimension
n_head = 4       # attention heads
n_layer = 1      # transformer layers
block_size = 8   # max sequence length
vocab_size = 27  # a-z + BOS token
num_params = 4064
```

## Running

```bash
python3 microgpt.py
```

No dependencies. No venv. Just Python stdlib.

## Training

- **Dataset**: 32k names (input.txt auto-downloads)
- **Task**: Character-level language modeling
- **Steps**: 500 training iterations
- **Loss**: Cross-entropy, ~3.3 → ~2.1
- **Optimizer**: Adam with cosine learning rate decay

## Output

```
sample 1: kalia
sample 2: ameli
sample 3: mayein
sample 4: jaylie
sample 5: karli
```

The model learns character patterns and generates plausible name-like sequences.

## Learning Path

Read the code top to bottom. Every line is documented. Key concepts:

1. **Tokenization** (lines 23-27): Characters become discrete tokens
2. **Autograd** (lines 29-72): Chain rule implementation for backprop
3. **Embeddings** (lines 108-111): Token + position encoding
4. **Attention** (lines 124-133): How tokens "look at" each other
5. **MLP** (lines 136-141): Feedforward transformation
6. **Softmax** (lines 97-101): Logits → probabilities
7. **Loss** (line 167): Negative log likelihood
8. **Backward** (line 172): Calculate all gradients
9. **Adam** (lines 175-182): Update parameters

## Why This Matters

Modern LLMs (GPT-4, Claude, etc.) use the same core principles:
- Tokenization → embeddings
- Multi-head self-attention
- Residual connections
- Layer normalization
- Autoregressive generation

This code strips away efficiency optimizations (GPU acceleration, batch processing, etc.) to expose the raw algorithm. Everything else is just scale.

## Credit

@karpathy - "Everything else is just efficiency"
