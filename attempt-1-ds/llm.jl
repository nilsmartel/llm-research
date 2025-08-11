using Flux
using Flux: onehot, onehotbatch, logitcrossentropy
using Statistics: mean

# Hyperparameters
const VOCAB_SIZE = 128
const EMBEDDING_DIM = 64
const HIDDEN_DIM = 128
const CONTEXT_LENGTH = 32
const BATCH_SIZE = 16

# Token embedding layer
struct TokenEmbedding
    embedding::Dense
end

TokenEmbedding(vocab_size, embed_dim) = TokenEmbedding(Dense(vocab_size => embed_dim, bias=false))

Flux.@functor TokenEmbedding

function (te::TokenEmbedding)(x)
    # x should be one-hot encoded tokens (vocab_size × sequence_length × batch_size)
    te.embedding(x)
end

# Simple self-attention layer
struct SelfAttention
    q::Dense
    k::Dense
    v::Dense
    output::Dense
end

SelfAttention(input_dim, head_dim) = SelfAttention(
    Dense(input_dim => head_dim, bias=false),
    Dense(input_dim => head_dim, bias=false),
    Dense(input_dim => head_dim, bias=false),
    Dense(head_dim => input_dim)
)

Flux.@functor SelfAttention

function (sa::SelfAttention)(x)
    # x shape: (embed_dim, seq_len, batch_size)
    q = sa.q(x)  # (head_dim, seq_len, batch_size)
    k = sa.k(x)  # (head_dim, seq_len, batch_size)
    v = sa.v(x)  # (head_dim, seq_len, batch_size)
    
    # Scaled dot-product attention
    scores = batched_mul(permutedims(q, (2,1,3)), k) ./ sqrt(size(q, 1))  # (seq_len, seq_len, batch_size)
    attn = softmax(scores; dims=1)
    out = batched_mul(v, attn)  # (head_dim, seq_len, batch_size)
    
    sa.output(out)
end

# Transformer block
struct TransformerBlock
    attention::SelfAttention
    mlp::Chain
    norm1::LayerNorm
    norm2::LayerNorm
end

TransformerBlock(embed_dim, head_dim, mlp_dim) = TransformerBlock(
    SelfAttention(embed_dim, head_dim),
    Chain(Dense(embed_dim => mlp_dim, gelu), Dense(mlp_dim => embed_dim)),
    LayerNorm(embed_dim),
    LayerNorm(embed_dim)
)

Flux.@functor TransformerBlock

function (tb::TransformerBlock)(x)
    # Self-attention with residual connection
    x = x + tb.attention(tb.norm1(x))
    # MLP with residual connection
    x = x + tb.mlp(tb.norm2(x))
    return x
end

# Simple Language Model
struct SimpleLLM
    token_embed::TokenEmbedding
    pos_embed::Dense
    blocks::Chain
    ln_f::LayerNorm
    head::Dense
end

function SimpleLLM(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, 
                  context_length=CONTEXT_LENGTH, n_layers=4)
    # Positional embeddings (simple learned embeddings)
    pos_embed = Dense(context_length => embed_dim, bias=false)
    
    # Transformer blocks
    blocks = Chain([TransformerBlock(embed_dim, embed_dim ÷ 2, hidden_dim) 
                   for _ in 1:n_layers]...)
    
    SimpleLLM(
        TokenEmbedding(vocab_size, embed_dim),
        pos_embed,
        blocks,
        LayerNorm(embed_dim),
        Dense(embed_dim => vocab_size)
    )
end

Flux.@functor SimpleLLM

function (m::SimpleLLM)(tokens)
    # tokens should be (sequence_length, batch_size) matrix of token indices
    
    # Create position indices
    seq_len, batch_size = size(tokens)
    pos_indices = collect(1:seq_len) .* ones(batch_size)'
    
    # Get token and position embeddings
    tok_emb = m.token_embed(onehotbatch(tokens, 1:VOCAB_SIZE))  # (embed_dim, seq_len, batch_size)
    pos_emb = m.pos_embed(onehotbatch(pos_indices, 1:CONTEXT_LENGTH))
    
    # Combine embeddings and pass through transformer
    x = tok_emb + pos_emb
    x = m.blocks(x)
    x = m.ln_f(x)
    
    # Prediction head
    logits = m.head(x)  # (vocab_size, seq_len, batch_size)
    
    return logits
end

# Training utilities
function get_batch(data, batch_size, context_length)
    # Generate random batches from data
    starts = rand(1:length(data)-context_length-1, batch_size)
    x = [data[start:start+context_length-1] for start in starts]
    y = [data[start+1:start+context_length] for start in starts]
    return hcat(x...), hcat(y...)
end

function train!(model, data; epochs=10, lr=3e-4)
    opt = Adam(lr)
    ps = Flux.params(model)
    
    for epoch in 1:epochs
        # Get batch
        x, y = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH)
        
        # Forward pass and loss calculation
        loss, grads = Flux.withgradient(ps) do
            logits = model(x)
            logitcrossentropy(logits, onehotbatch(y, 1:VOCAB_SIZE))
        end
        
        # Update parameters
        Flux.update!(opt, ps, grads)
        
        if epoch % 10 == 0
            println("Epoch $epoch, Loss: $(loss)")
        end
    end
end

# Example usage
function main()
    # Create model
    model = SimpleLLM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, CONTEXT_LENGTH)
    
    # Generate random training data (in practice, you'd use real text)
    data = rand(1:VOCAB_SIZE, 10000)
    
    # Train the model
    train!(model, data, epochs=100)
    
    # Generate some text
    function generate(model, start_tokens, length=50)
        tokens = copy(start_tokens)
        for _ in 1:length
            context = tokens[end-min(length, CONTEXT_LENGTH)+1:end]
            logits = model(reshape(context, length(context), 1))
            next_token = argmax(logits[:, end, 1])
            push!(tokens, next_token)
        end
        return tokens
    end
    
    start_tokens = rand(1:VOCAB_SIZE, 5)
    generated = generate(model, start_tokens)
    println("Generated sequence: ", generated)
end

# Uncomment to run
# main()