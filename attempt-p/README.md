# Simple LLM in Julia (Character-level LSTM, vocab=128)

This is a minimal character-level language model implemented in Julia using Flux.jl. It operates over a fixed ASCII vocabulary of size 128 (characters 0..127), and can train on your text and generate new text.

Features:
- Tokenization to 128-sized ASCII vocabulary
- Embedding -> LSTM -> Linear projection over tokens
- Cross-entropy training
- Text generation with temperature and top-k sampling

## Requirements

- Julia 1.9+
- Flux.jl

Install Flux:

```julia
using Pkg
Pkg.add("Flux")
```

## Usage

Optional: place your training data at `data/input.txt` (ASCII recommended). If not provided, a small built-in sample is used.

Run:

```bash
julia --project=. simple_llm.jl
```

You can override hyperparameters via environment variables:

- `D_MODEL` (default 256): embedding size
- `HIDDEN` (default 512): LSTM hidden size
- `SEQ_LEN` (default 128): unroll length for BPTT
- `BATCH_SIZE` (default 32)
- `EPOCHS` (default 5)
- `STEPS_PER_EPOCH` (default 100)
- `LR` (default 0.001)
- `SEED` (default 42)
- `DATA_PATH` (default `data/input.txt`)

Example:

```bash
D_MODEL=384 HIDDEN=768 EPOCHS=10 STEPS_PER_EPOCH=200 DATA_PATH=data/input.txt julia simple_llm.jl
```

## Notes

- Non-ASCII characters are mapped to space during tokenization.
- This is a simple baseline. For larger-scale models or transformer-based decoders (GPT-style), consider using multi-head self-attention blocks and causal masking.
- To speed up training, you can reduce `SEQ_LEN`, `BATCH_SIZE`, or `STEPS_PER_EPOCH`.