# How LLMs Work

An interactive app that walks through every stage of the LLM pipeline — from pattern matching to training a transformer from scratch — with working code you can run locally.

- [How LLMs Work](#how-llms-work)
  - [Stack](#stack)
  - [Setup](#setup)
  - [Sections](#sections)
    - [1. Simple Chat — Pattern Matching](#1-simple-chat--pattern-matching)
    - [2. XOR Neural Net — Backpropagation](#2-xor-neural-net--backpropagation)
    - [3. Basic Tokenizer — BPE From Scratch](#3-basic-tokenizer--bpe-from-scratch)
    - [4. Train Embeddings — Word2Vec Skip-Gram](#4-train-embeddings--word2vec-skip-gram)
    - [5. Train Transformer — GPT From Scratch](#5-train-transformer--gpt-from-scratch)
  - [Architecture](#architecture)
  - [References](#references)

## Stack

- [Hono](https://hono.dev) — server, JSX rendering, and client components
- [Vite](https://vite.dev) — dev server + bundling

## Setup

Requires [pnpm](https://pnpm.io/installation) and Node.js 20+.

```sh
pnpm install
pnpm dev
```

## Sections

### 1. Simple Chat — Pattern Matching

The simplest possible "AI": an ELIZA-style chatbot that responds with if-statements, streamed word by word via SSE. Same plumbing as ChatGPT, zero intelligence.

- Route: [`src/routes/simple-chat.ts`](src/routes/simple-chat.ts)
- Hook: [`src/client/hooks/use-simple-chat.ts`](src/client/hooks/use-simple-chat.ts)

### 2. XOR Neural Net — Backpropagation

Trains a neural network live. A single-layer perceptron fails on XOR (proving Minsky/Papert right). A multi-layer network succeeds via backpropagation — the same algorithm every neural network uses today.

- Route: [`src/routes/neural-net/`](src/routes/neural-net/)
- Hook: [`src/client/hooks/use-neural-net-chat.tsx`](src/client/hooks/use-neural-net-chat.tsx)
- Component: [`src/client/components/neural-net-result/`](src/client/components/neural-net-result/)

### 3. Basic Tokenizer — BPE From Scratch

A from-scratch BPE implementation that trains on your input text. Watch merge steps animate as the algorithm builds a vocabulary from characters to words.

- Route: [`src/routes/bpe-tokenize.ts`](src/routes/bpe-tokenize.ts)
- Hook: [`src/client/hooks/use-bpe-tokenize-chat.tsx`](src/client/hooks/use-bpe-tokenize-chat.tsx)
- Component: [`src/client/components/bpe-tokenize-result/`](src/client/components/bpe-tokenize-result/)

### 4. Train Embeddings — Word2Vec Skip-Gram

Trains word embeddings from scratch using Word2Vec skip-gram with negative sampling. Watch vectors learn that words used in similar contexts should cluster together.

- Route: [`src/routes/train-embed/`](src/routes/train-embed/)
- Hook: [`src/client/hooks/use-train-embed-chat.tsx`](src/client/hooks/use-train-embed-chat.tsx)
- Component: [`src/client/components/train-embed-result/`](src/client/components/train-embed-result/)

### 5. Train Transformer — GPT From Scratch

Trains a decoder-only transformer entirely from scratch — no ML libraries. Every operation is implemented by hand: multi-head causal self-attention, layer normalization, feed-forward layers, backpropagation, and Adam optimization. Uses multi-threaded data parallelism via SharedArrayBuffer for training speed.

- Route: [`src/routes/train-transformer/`](src/routes/train-transformer/)
- Hook: [`src/client/hooks/use-train-transformer-chat.tsx`](src/client/hooks/use-train-transformer-chat.tsx)
- Component: [`src/client/components/train-transformer-result/`](src/client/components/train-transformer-result/)

## Architecture

Every section follows the same pattern:

1. **Server route** (`src/routes/`) — Hono POST handler that processes input and streams SSE events
2. **Client hook** (`src/client/hooks/`) — manages state and connects SSE events to UI updates via `useSSEChat`
3. **Result component** (`src/client/components/`) — renders the streamed data as a visualization

Core infrastructure:

- SSE streaming: [`src/server/lib/sse.ts`](src/server/lib/sse.ts) (server) / [`src/client/lib/sse.ts`](src/client/lib/sse.ts) (client)
- BPE tokenizer: [`src/server/lib/bpe.ts`](src/server/lib/bpe.ts) (shared by Basic Tokenizer, Train Embeddings, and Train Transformer)
- Generic chat hook: [`src/client/hooks/use-sse-chat.ts`](src/client/hooks/use-sse-chat.ts)
- Message types: [`src/shared/types/message.ts`](src/shared/types/message.ts)

## References

Papers referenced in the codebase, in the order the concepts appear across the demos:

1. **Weizenbaum (1966)** — "ELIZA — A Computer Program for the Study of Natural Language Communication Between Man and Machine"
   [dl.acm.org/doi/10.1145/365153.365168](https://dl.acm.org/doi/10.1145/365153.365168)
   _Section 1 — the original pattern-matching chatbot that inspired the Simple Chat demo._

2. **Rumelhart, Hinton & Williams (1986)** — "Learning Representations by Back-Propagating Errors"
   [nature.com/articles/323533a0](https://www.nature.com/articles/323533a0)
   _Section 2 — the backpropagation algorithm that made multi-layer neural networks trainable._

3. **Glorot & Bengio (2010)** — "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
   [proceedings.mlr.press/v9/glorot10a.html](https://proceedings.mlr.press/v9/glorot10a.html)
   _Section 5 — Xavier/Glorot initialization, used to set initial transformer weights._

4. **Mikolov et al. (2013a)** — "Efficient Estimation of Word Representations in Vector Space"
   [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
   _Section 4 — introduces Word2Vec and the Skip-gram architecture used in the embedding trainer._

5. **Mikolov et al. (2013b)** — "Distributed Representations of Words and Phrases and their Compositionality"
   [arxiv.org/abs/1310.4546](https://arxiv.org/abs/1310.4546)
   _Section 4 — introduces negative sampling, the training trick that makes Skip-gram practical._

6. **Kingma & Ba (2014)** — "Adam: A Method for Stochastic Optimization"
   [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
   _Section 5 — the Adam optimizer used to train the transformer._

7. **Sennrich, Haddow & Birch (2016)** — "Neural Machine Translation of Rare Words with Subword Units"
   [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
   _Sections 3, 4, 5 — Byte Pair Encoding (BPE), the tokenization algorithm used throughout._

8. **Ba, Kiros & Hinton (2016)** — "Layer Normalization"
   [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
   _Section 5 — layer normalization, applied before attention and feed-forward layers in the transformer._

9. **Vaswani et al. (2017)** — "Attention Is All You Need"
   [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   _Section 5 — the transformer architecture implemented from scratch._

10. **Radford et al. (2018)** — "Improving Language Understanding by Generative Pre-Training"
    [cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    _Section 5 — GPT-1, the decoder-only transformer pre-training approach this demo follows._

11. **Holtzman et al. (2019)** — "The Curious Case of Neural Text Degeneration"
    [arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)
    _Section 5 — nucleus (top-p) sampling, used for text generation after training._
