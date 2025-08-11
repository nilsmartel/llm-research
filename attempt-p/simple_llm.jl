# Simple character-level LLM in Julia (LSTM-based) with vocab size 128
#
# - Trains on ASCII text (characters with code <= 127)
# - Embedding -> LSTM -> Linear projection to logits over 128 tokens
# - Generates text with temperature and optional top-k sampling
#
# Usage:
#   julia --project=. simple_llm.jl
#
# Dependencies:
#   using Pkg
#   Pkg.add("Flux")
#
# Optional:
#   Put your training data at data/input.txt (ASCII recommended).
#   Otherwise, a small built-in sample is used.

using Random
using Statistics
using Flux

# ----------------------------
# Tokenization (ASCII 0..127 -> ids 1..128)
# ----------------------------

const VOCAB_SIZE = 128

# Map ASCII char to id in 1..128, fallback non-ASCII -> space
ascii_to_id(c::Char) = (Int(c) <= 0x7F ? Int(c) + 1 : Int(' ') + 1)
id_to_ascii(id::Int) = Char(clamp(id, 1, VOCAB_SIZE) - 1)

function text_to_ids(text::AbstractString)
    ids = Vector{Int}(undef, ncodeunits(text))  # worst-case allocation
    j = 0
    for c in text
        j += 1
        ids[j] = ascii_to_id(c)
    end
    return @view(ids[1:j]) |> collect
end

function ids_to_text(ids::AbstractVector{<:Integer})
    IOBuffer() do io
        for id in ids
            print(io, id_to_ascii(id))
        end
        String(take!(io))
    end
end

# ----------------------------
# Model
# ----------------------------

struct SimpleRNNLM
    embed::Flux.Embed
    rnn::Flux.Recur{Flux.LSTMCell}
    proj::Dense
    vocab::Int
end

function SimpleRNNLM(vocab::Int; d_model::Int=256, hidden::Int=512)
    embed = Flux.Embed(vocab, d_model)         # d_model × batch
    rnn   = Flux.Recur(Flux.LSTM(d_model, hidden))
    proj  = Dense(hidden, vocab)               # vocab × batch
    SimpleRNNLM(embed, rnn, proj, vocab)
end

Flux.@functor SimpleRNNLM

# Forward over one step: input ids vector -> logits (vocab × batch)
function step_logits!(m::SimpleRNNLM, x_ids::AbstractVector{<:Integer})
    x_emb = m.embed(x_ids)      # (d_model × batch)
    h     = m.rnn(x_emb)        # (hidden × batch)
    m.proj(h)                   # (vocab × batch)
end

# Loss over a full sequence batch (seq_len × batch)
function seq_ce_loss!(m::SimpleRNNLM, X::AbstractMatrix{<:Integer}, Y::AbstractMatrix{<:Integer})
    Flux.reset!(m.rnn)
    T, B = size(X)
    loss = 0.0f0
    for t in 1:T
        logits = step_logits!(m, @view X[t, :])                # (vocab × B)
        y_oh   = Flux.onehotbatch(@view Y[t, :], 1:m.vocab)    # (vocab × B)
        loss  += Flux.logitcrossentropy(logits, y_oh)
    end
    loss / T
end

# ----------------------------
# Data loader
# ----------------------------

# Create random mini-batches (seq_len × batch) from a flat ids array
function get_batch(ids::Vector{Int}, seq_len::Int, batch_size::Int)
    N = length(ids)
    @assert N > seq_len + 1 "Not enough data to create a batch"
    X = Array{Int}(undef, seq_len, batch_size)
    Y = Array{Int}(undef, seq_len, batch_size)
    for b in 1:batch_size
        start = rand(1:(N - seq_len - 1))
        X[:, b] = @view ids[start:start+seq_len-1]
        Y[:, b] = @view ids[start+1:start+seq_len]
    end
    return X, Y
end

# ----------------------------
# Training
# ----------------------------

mutable struct TrainConfig
    epochs::Int
    steps_per_epoch::Int
    seq_len::Int
    batch_size::Int
    lr::Float64
    seed::Int
end

function train!(m::SimpleRNNLM, ids::Vector{Int}, cfg::TrainConfig)
    Random.seed!(cfg.seed)
    ps  = Flux.params(m)
    opt = Flux.Adam(cfg.lr)

    for epoch in 1:cfg.epochs
        epoch_loss = 0.0
        for step in 1:cfg.steps_per_epoch
            X, Y = get_batch(ids, cfg.seq_len, cfg.batch_size)
            grads = Flux.gradient(ps) do
                seq_ce_loss!(m, X, Y)
            end
            Flux.Optimise.update!(opt, ps, grads)
            loss = seq_ce_loss!(m, X, Y) |> float
            epoch_loss += loss
            if step % max(1, cfg.steps_per_epoch ÷ 5) == 0
                @info "epoch=$epoch step=$step loss=$(round(loss, digits=4))"
            end
        end
        @info "Epoch $epoch avg_loss=$(round(epoch_loss / cfg.steps_per_epoch, digits=4))"
    end
end

# ----------------------------
# Sampling
# ----------------------------

# Sample next token id from logits vector (vocab × 1), with temperature and optional top-k
function sample_id_from_logits(logits::AbstractVector{<:Real}; temperature::Float64=1.0, top_k::Int=0)
    z = collect(logits) ./ max(1e-6, temperature)
    if top_k > 0 && top_k < length(z)
        idxs = partialsortperm(z, 1:top_k; rev=true)
        mask = trues(length(z))
        mask[idxs] .= false
        # Set masked logits to -Inf
        for (i, m) in enumerate(mask)
            if m; z[i] = -Inf; end
        end
    end
    p = Flux.softmax(z)
    r = rand()
    s = 0.0
    for i in eachindex(p)
        s += p[i]
        if r <= s
            return i
        end
    end
    return length(p) # fallback
end

function generate(m::SimpleRNNLM, start_text::AbstractString, n_tokens::Int; temperature::Float64=1.0, top_k::Int=0)
    # Prime the RNN with start_text
    Flux.reset!(m.rnn)
    start_ids = text_to_ids(start_text)
    for id in start_ids
        _ = step_logits!(m, [id])
    end
    out_ids = Int[]
    last_id = isempty(start_ids) ? ascii_to_id(' ') : last(start_ids)
    for _ in 1:n_tokens
        logits = vec(step_logits!(m, [last_id]))  # (vocab,)
        next_id = sample_id_from_logits(logits; temperature=temperature, top_k=top_k)
        push!(out_ids, next_id)
        last_id = next_id
    end
    start_text * ids_to_text(out_ids)
end

# ----------------------------
# Main
# ----------------------------

function maybe_read_data(path::AbstractString)
    if isfile(path)
        open(path, "r") do io
            read(io, String)
        end
    else
        # Fallback tiny sample text (ASCII)
        """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them.
        """
    end
end

function main()
    # Hyperparameters
    d_model   = parse(Int, get(ENV, "D_MODEL", "256"))
    hidden    = parse(Int, get(ENV, "HIDDEN", "512"))
    seq_len   = parse(Int, get(ENV, "SEQ_LEN", "128"))
    batch_sz  = parse(Int, get(ENV, "BATCH_SIZE", "32"))
    epochs    = parse(Int, get(ENV, "EPOCHS", "5"))
    steps_ep  = parse(Int, get(ENV, "STEPS_PER_EPOCH", "100"))
    lr        = parse(Float64, get(ENV, "LR", "0.001"))
    seed      = parse(Int, get(ENV, "SEED", "42"))
    data_path = get(ENV, "DATA_PATH", "data/input.txt")

    println("Loading data from: $data_path")
    text = maybe_read_data(data_path)
    ids  = text_to_ids(text)
    println("Dataset size (tokens): ", length(ids))

    println("Building model: vocab=$VOCAB_SIZE d_model=$d_model hidden=$hidden")
    model = SimpleRNNLM(VOCAB_SIZE; d_model=d_model, hidden=hidden)

    cfg = TrainConfig(epochs, steps_ep, seq_len, batch_sz, lr, seed)
    train!(model, ids, cfg)

    println("\n--- Generation ---")
    prompt = "To be, or not to be"
    gen = generate(model, prompt, 400; temperature=0.9, top_k=20)
    println(gen)
end

abspath(PROGRAM_FILE) == @__FILE__ && main()