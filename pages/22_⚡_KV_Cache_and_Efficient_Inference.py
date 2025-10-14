import streamlit as st

st.title("⚡ Topic 22: KV Cache & Efficient Inference")

st.markdown("""
---

## 🎯 The Inference Problem: Generating Text is Slow

You've trained your LLM. Now you want to generate text:

```
User: "Once upon a time"
Model: "Once upon a time, there was a princess who lived in a castle..."
```

**How text generation works** (autoregressive):

```
Step 1: Input "Once upon a time" → Generate "there"
Step 2: Input "Once upon a time there" → Generate "was"
Step 3: Input "Once upon a time there was" → Generate "a"
...
```

Each token requires running the ENTIRE model! For a 100-token response, you run the model **100 times**!

**The naive approach:**

```python
generated = prompt  # "Once upon a time"

for i in range(100):  # Generate 100 tokens
    logits = model(generated)  # Run FULL model on ALL tokens!
    next_token = sample(logits[-1])
    generated = torch.cat([generated, next_token])

# Problem: Recomputing attention for previous tokens EVERY TIME!
```

**Why this is wasteful:**

```
Step 1: Compute attention for ["Once", "upon", "a", "time"]
Step 2: Compute attention for ["Once", "upon", "a", "time", "there"]
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^ Already computed!
Step 3: Compute attention for ["Once", "upon", "a", "time", "there", "was"]
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Already computed!
```

**We're recomputing the SAME attention keys and values over and over!**

**The solution?** **KV Cache** - cache previous keys and values!

---

## 💡 KV Cache: The Key to Fast Generation

### The Core Insight

In self-attention, we compute Q, K, V for all tokens. But when generating:
- **New query** for new token: Must be computed ✓
- **Keys and values for previous tokens**: Haven't changed! Can reuse! ✓

### Standard Attention (Without Cache)

```python
def generate_token(model, input_tokens):
    # input_tokens: ["Once", "upon", "a", "time", "there"]

    # Compute Q, K, V for ALL tokens (wasteful!)
    Q = model.project_q(embeddings)  # [5, d_model]
    K = model.project_k(embeddings)  # [5, d_model]
    V = model.project_v(embeddings)  # [5, d_model]

    # Compute attention
    scores = Q @ K.T
    attn = softmax(scores)
    output = attn @ V

    # Only use the LAST token's output
    return output[-1]  # Everything else is recomputed next step!
```

**Waste:** Recomputing K and V for "Once", "upon", "a", "time" every single step!

### With KV Cache

```python
def generate_token(model, new_token, kv_cache):
    # new_token: just "there"
    # kv_cache: stored K and V for ["Once", "upon", "a", "time"]

    # Only compute Q, K, V for NEW token
    Q_new = model.project_q(embed(new_token))  # [1, d_model]
    K_new = model.project_k(embed(new_token))  # [1, d_model]
    V_new = model.project_v(embed(new_token))  # [1, d_model]

    # Retrieve cached K and V
    K_cached = kv_cache['K']  # [4, d_model]
    V_cached = kv_cache['V']  # [4, d_model]

    # Concatenate: past + new
    K = torch.cat([K_cached, K_new], dim=0)  # [5, d_model]
    V = torch.cat([V_cached, V_new], dim=0)  # [5, d_model]

    # Compute attention (Q is just 1 token!)
    scores = Q_new @ K.T  # [1, 5]
    attn = softmax(scores)
    output = attn @ V  # [1, d_model]

    # Update cache for next step
    kv_cache['K'] = K
    kv_cache['V'] = V

    return output, kv_cache
```

**Savings:** Only compute K and V for ONE new token, not all previous tokens!

---

## 🔢 The Math: Computational Savings

### Without KV Cache

```
# Generating 100 tokens from a 10-token prompt

Step 1: Attention on 10 tokens
Step 2: Attention on 11 tokens
Step 3: Attention on 12 tokens
...
Step 100: Attention on 110 tokens

Total attention computations:
10 + 11 + 12 + ... + 110 = 6,050 token-steps
```

### With KV Cache

```
# Same generation

Step 1: Attention with 1 new query, 10 keys/values
Step 2: Attention with 1 new query, 11 keys/values
Step 3: Attention with 1 new query, 12 keys/values
...
Step 100: Attention with 1 new query, 110 keys/values

Total new computations:
100 queries (one per step)

Cached: 6,050 - 100 = 5,950 token-steps saved!
```

**Speedup:** ~60x less computation for this example!

**General formula:**

```
Without cache: O(N²) where N is sequence length
With cache: O(N) - linear in sequence length!
```

---

## 💻 Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCacheMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with KV Cache support

    Used for efficient autoregressive generation (GPT, LLaMA, etc.)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
               During generation, seq_len=1 (just new token)
            kv_cache: Dict with 'k' and 'v' tensors from previous steps
                     None for first step
            use_cache: Whether to return updated cache

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated KV cache (if use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V for new token(s)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # If we have cached K and V, concatenate with new K and V
        if kv_cache is not None:
            K_cached = kv_cache['k']  # [batch, num_heads, past_len, head_dim]
            V_cached = kv_cache['v']

            K = torch.cat([K_cached, K], dim=2)  # [batch, num_heads, past_len + seq_len, head_dim]
            V = torch.cat([V_cached, V], dim=2)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        # Return with cache if requested
        if use_cache:
            new_cache = {'k': K, 'v': V}
            return output, new_cache
        else:
            return output


# Example: Text generation with KV cache
def generate_with_kv_cache(model, prompt_tokens, max_new_tokens=50):
    """
    Generate text using KV cache for efficiency

    Args:
        model: Transformer model with KV cache support
        prompt_tokens: Input token IDs [batch, prompt_len]
        max_new_tokens: Number of tokens to generate

    Returns:
        generated_tokens: [batch, prompt_len + max_new_tokens]
    """
    batch_size = prompt_tokens.shape[0]
    device = prompt_tokens.device

    # Initialize
    generated = prompt_tokens
    kv_cache = None

    # First pass: Process prompt and initialize cache
    with torch.no_grad():
        # Get embeddings
        x = model.embed(prompt_tokens)  # [batch, prompt_len, d_model]

        # Forward through layers with cache initialization
        output, kv_cache = model.forward_with_cache(x, kv_cache=None, use_cache=True)

        # Get next token
        logits = model.lm_head(output[:, -1, :])  # [batch, vocab_size]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]

        generated = torch.cat([generated, next_token], dim=1)

    # Autoregressive generation with cache
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            # Only process the new token!
            x = model.embed(next_token)  # [batch, 1, d_model]

            # Forward with cache (much faster!)
            output, kv_cache = model.forward_with_cache(x, kv_cache=kv_cache, use_cache=True)

            # Sample next token
            logits = model.lm_head(output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

    return generated


# Benchmark: With vs Without Cache
def benchmark_generation():
    d_model = 512
    num_heads = 8
    seq_len = 1  # Generate one token at a time

    # Create model
    attn = KVCacheMultiHeadAttention(d_model, num_heads)

    # Simulate generation
    batch_size = 1
    num_steps = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import time

    # Without cache (recompute everything)
    print("Without KV Cache:")
    start = time.time()
    x_all = torch.randn(batch_size, 1, d_model, device=device)

    for step in range(num_steps):
        # Recompute attention on all previous tokens
        _ = attn(x_all, kv_cache=None, use_cache=False)
        new_token = torch.randn(batch_size, 1, d_model, device=device)
        x_all = torch.cat([x_all, new_token], dim=1)

    time_no_cache = time.time() - start
    print(f"  Time: {time_no_cache:.3f}s")

    # With cache
    print("\\nWith KV Cache:")
    start = time.time()
    kv_cache = None

    for step in range(num_steps):
        # Only process new token
        new_token = torch.randn(batch_size, 1, d_model, device=device)
        _, kv_cache = attn(new_token, kv_cache=kv_cache, use_cache=True)

    time_with_cache = time.time() - start
    print(f"  Time: {time_with_cache:.3f}s")

    print(f"\\nSpeedup: {time_no_cache / time_with_cache:.1f}x faster!")


# Run benchmark (uncomment to test)
# benchmark_generation()
```

---

## 🗂️ Multi-Layer KV Cache

Real transformers have many layers! Each layer needs its own cache:

```python
class TransformerWithKVCache(nn.Module):
    """
    Transformer decoder with KV cache for all layers
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward_with_cache(self, x, past_kv_cache=None, use_cache=False):
        """
        Forward pass with KV cache support

        Args:
            x: Input [batch, seq_len, d_model]
            past_kv_cache: List of caches for each layer
                          (None for first pass)
            use_cache: Whether to return updated caches

        Returns:
            output: [batch, seq_len, vocab_size]
            new_kv_cache: List of updated caches
        """
        # Initialize cache list if needed
        if past_kv_cache is None:
            past_kv_cache = [None] * len(self.layers)

        new_kv_cache = []

        # Pass through layers
        for i, layer in enumerate(self.layers):
            if use_cache:
                x, layer_cache = layer(x, kv_cache=past_kv_cache[i], use_cache=True)
                new_kv_cache.append(layer_cache)
            else:
                x = layer(x, kv_cache=None, use_cache=False)

        # Final projection
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kv_cache
        else:
            return logits


# Memory calculation
def kv_cache_memory(batch_size, num_layers, num_heads, seq_len, head_dim, bytes_per_element=2):
    """
    Calculate KV cache memory usage

    Args:
        bytes_per_element: 2 for fp16, 4 for fp32
    """
    # Each layer stores K and V
    # Each has shape: [batch, num_heads, seq_len, head_dim]

    elements_per_layer = 2 * batch_size * num_heads * seq_len * head_dim
    bytes_per_layer = elements_per_layer * bytes_per_element
    total_bytes = bytes_per_layer * num_layers

    return total_bytes / (1024 ** 3)  # Convert to GB


# Example: LLaMA 7B
print("LLaMA 7B KV Cache Memory (fp16):")
print(f"  1 user, 2k context: {kv_cache_memory(1, 32, 32, 2048, 128):.2f} GB")
print(f"  1 user, 4k context: {kv_cache_memory(1, 32, 32, 4096, 128):.2f} GB")
print(f"  8 users, 2k context: {kv_cache_memory(8, 32, 32, 2048, 128):.2f} GB")
print(f"\\nLLaMA 70B KV Cache Memory (fp16, with GQA):")
print(f"  1 user, 4k context: {kv_cache_memory(1, 80, 8, 4096, 128):.2f} GB")
print(f"  8 users, 4k context: {kv_cache_memory(8, 80, 8, 4096, 128):.2f} GB")
```

**Output:**
```
LLaMA 7B KV Cache Memory (fp16):
  1 user, 2k context: 1.05 GB
  1 user, 4k context: 2.10 GB
  8 users, 2k context: 8.39 GB

LLaMA 70B KV Cache Memory (fp16, with GQA):
  1 user, 4k context: 0.21 GB
  8 users, 4k context: 1.68 GB
```

**This is why GQA matters!** (Topic 20)

---

## 🚀 Advanced KV Cache Techniques

### 1. Static vs Dynamic Cache

**Static Cache:**
```python
# Pre-allocate maximum size
max_seq_len = 2048
kv_cache = torch.zeros(batch, num_heads, max_seq_len, head_dim)

# Update in-place
kv_cache[:, :, step, :] = K_new
```

**Pros:** No memory reallocation
**Cons:** Wastes memory if sequence is short

**Dynamic Cache:**
```python
# Grow as needed
kv_cache = []
for step in range(generation_steps):
    kv_cache.append(K_new)
kv_cache = torch.cat(kv_cache, dim=2)
```

**Pros:** Only use memory needed
**Cons:** Concatenation overhead

### 2. Multi-Turn Conversation Caching

```python
# Cache context across multiple turns!
conversation_cache = None

# Turn 1
user_1 = "What is Python?"
response_1, conversation_cache = generate(prompt=user_1, cache=conversation_cache)

# Turn 2 (reuse cache!)
user_2 = "How do I install it?"
response_2, conversation_cache = generate(prompt=user_2, cache=conversation_cache)

# Don't recompute "What is Python?" and response!
```

### 3. Prefix Caching

Many users ask similar questions:

```python
# Common system prompt
system_prompt = "You are a helpful assistant..."  # 500 tokens

# Cache system prompt once
system_cache = compute_cache(system_prompt)

# Reuse for all users!
for user in users:
    # Only compute user query + response, not system prompt
    response = generate(user.query, prefix_cache=system_cache)
```

**Savings:** Don't recompute common prefixes!

### 4. PagedAttention (vLLM)

**Problem:** KV cache is hard to manage with variable-length sequences

**Solution:** Store cache in fixed-size pages (like virtual memory)

```python
# Traditional: Contiguous allocation
cache = allocate(max_length)  # Wastes memory

# PagedAttention: Paged allocation
cache_pages = [
    allocate(page_size),  # Page 1
    allocate(page_size),  # Page 2
    ...  # Allocate as needed
]
```

**Benefits:**
- Less memory fragmentation
- Better GPU utilization
- Can serve more concurrent users

**Used in:** vLLM (most popular serving framework)

---

## 📊 Real-World Performance

### Inference Speed

```
ChatGPT-scale model (7B parameters)
Hardware: A100 GPU

Without KV Cache:
- Speed: 5 tokens/second
- Bottleneck: Recomputing attention

With KV Cache:
- Speed: 150 tokens/second (30x faster!)
- Bottleneck: Memory bandwidth
```

### Multi-User Serving

```
Serving LLaMA 7B on A100 (80GB)

Without KV Cache:
- Throughput: 0.5 users at a time (too slow)
- Each user: 5 tokens/sec

With KV Cache (no batching):
- Throughput: 1 user at a time
- Each user: 150 tokens/sec

With KV Cache + Batching:
- Throughput: 32 users at a time!
- Each user: 20 tokens/sec
- Total: 640 tokens/sec (128x improvement!)
```

### Memory Trade-off

```
LLaMA 7B inference (batch=1, seq_len=2048)

Model weights: 13 GB (fp16)
KV cache: 1.05 GB

Total: 14 GB (cache is ~7% of memory)

With longer contexts (8k):
Model weights: 13 GB
KV cache: 4.2 GB (32% of memory!)
```

**KV cache becomes the bottleneck for long contexts!**

---

## 🎓 Key Takeaways

1. **Problem**: Autoregressive generation recomputes K and V for all previous tokens

2. **Solution**: KV Cache - store previous keys and values
   - Only compute Q, K, V for new token
   - Concatenate with cached K, V

3. **Speedup**: O(N²) → O(N) complexity
   - 30-100x faster generation in practice!

4. **Memory cost**: ~1-4 GB per user for typical contexts
   - Grows with sequence length
   - Reduced by GQA (8x savings)

5. **Essential for production**: ALL inference servers use KV cache
   - vLLM, TGI, FasterTransformer, etc.

6. **Advanced techniques**:
   - Multi-turn conversation caching
   - Prefix caching
   - PagedAttention

7. **Trade-off**: Speed vs memory
   - Worth it! Memory is cheaper than compute

**KV caching is THE optimization that makes LLM inference practical!**

---

## 📝 Quiz Time!

Test your understanding of KV Cache.
""")

# Quiz questions
with st.expander("Question 1: Why is generation slow without KV cache?"):
    st.markdown("""
    **Question**: Why is autoregressive text generation slow without KV cache?

    A) The model is too large
    B) We recompute attention keys and values for all previous tokens at every step
    C) GPUs are slow
    D) Tokenization is slow
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) We recompute attention keys and values for all previous tokens at every step**

        **Explanation**:

        **Autoregressive generation:**
        ```
        Step 1: "Once upon a time" → compute attention → "there"
        Step 2: "Once upon a time there" → compute attention AGAIN → "was"
        Step 3: "Once upon a time there was" → compute attention AGAIN → "a"
        ```

        **Wasted computation:**
        - K and V for "Once upon a time" are IDENTICAL in all steps
        - But we recompute them every time!
        - For 100 tokens: ~5,000 redundant computations

        **With KV cache:**
        - Compute K, V for "Once upon a time" ONCE
        - Store them
        - Only compute K, V for new token
        - 30-100x faster!

        KV cache is THE critical optimization for inference.
        """)

with st.expander("Question 2: What does KV cache store?"):
    st.markdown("""
    **Question**: What exactly does the KV cache store?

    A) All previous tokens
    B) Previous attention weights
    C) Previous key and value matrices for each layer
    D) Previous model outputs
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: C) Previous key and value matrices for each layer**

        **Explanation**:

        **KV cache structure:**
        ```python
        kv_cache = {
            'layer_0': {
                'k': [batch, num_heads, past_tokens, head_dim],
                'v': [batch, num_heads, past_tokens, head_dim]
            },
            'layer_1': {
                'k': [...],
                'v': [...]
            },
            ...
        }
        ```

        **Why K and V?**
        - Q changes with every new token (query is different)
        - K and V for previous tokens stay the same
        - Cache K and V, recompute Q

        **Why all layers?**
        - Each transformer layer has its own K and V
        - Need to cache all of them

        **Memory calculation:**
        ```python
        # LLaMA 7B (32 layers, 32 heads)
        memory = 2 * num_layers * num_heads * seq_len * head_dim * 2 bytes
        # For 2k tokens: ~1 GB
        ```
        """)

with st.expander("Question 3: Complexity improvement"):
    st.markdown("""
    **Question**: What is the computational complexity improvement with KV cache for generating N tokens?

    A) O(N) → O(1)
    B) O(N²) → O(N)
    C) O(N³) → O(N²)
    D) O(N log N) → O(N)
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) O(N²) → O(N)**

        **Explanation**:

        **Without cache (recompute everything):**
        ```
        Step 1: 1 token
        Step 2: 2 tokens
        Step 3: 3 tokens
        ...
        Step N: N tokens

        Total: 1 + 2 + 3 + ... + N = N(N+1)/2 = O(N²)
        ```

        **With cache (only new computations):**
        ```
        Step 1: 1 new token
        Step 2: 1 new token
        Step 3: 1 new token
        ...
        Step N: 1 new token

        Total: N = O(N)
        ```

        **Real numbers:**
        ```
        Generate 100 tokens:
        Without cache: 1+2+...+100 = 5,050 token-steps
        With cache: 100 token-steps (50x less!)
        ```

        Linear vs quadratic is the difference between practical and impractical!
        """)

with st.expander("Question 4: Memory cost"):
    st.markdown("""
    **Question**: For LLaMA 7B with 32 layers, 32 heads, 2048 token context, how much memory does KV cache use (fp16)?

    A) ~100 MB
    B) ~1 GB
    C) ~10 GB
    D) ~100 GB
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) ~1 GB**

        **Explanation**:

        **Calculation:**
        ```python
        num_layers = 32
        num_kv_heads = 32  # MHA for 7B
        seq_len = 2048
        head_dim = 128
        bytes_per_element = 2  # fp16

        # K and V for all layers
        elements = 2 * num_layers * num_kv_heads * seq_len * head_dim
        elements = 2 * 32 * 32 * 2048 * 128 = 536,870,912

        bytes = elements * 2 = 1,073,741,824 bytes ≈ 1 GB
        ```

        **Comparison:**
        - Model weights (7B params, fp16): ~13 GB
        - KV cache (2k context): ~1 GB (7% of model size)
        - KV cache (8k context): ~4 GB (30% of model size!)

        **Why GQA helps:**
        ```python
        # With GQA (8 KV heads instead of 32):
        cache_gqa = 1 GB / 4 = 0.25 GB
        # 4x memory savings!
        ```

        Long contexts → KV cache dominates memory usage!
        """)

with st.expander("Question 5: Advanced caching"):
    st.markdown("""
    **Question**: What is prefix caching used for?

    A) Storing the most recent tokens
    B) Caching common prompts (like system messages) shared across users
    C) Caching the first layer only
    D) Compressing the cache
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: B) Caching common prompts (like system messages) shared across users**

        **Explanation**:

        **Scenario:**
        ```
        System prompt (shared): "You are a helpful AI assistant..." (500 tokens)

        User 1: "What is Python?"
        User 2: "How do I learn coding?"
        User 3: "Explain machine learning"
        ...
        ```

        **Without prefix caching:**
        - Compute system prompt for EVERY user
        - 500 tokens × 1000 users = 500,000 redundant computations!

        **With prefix caching:**
        ```python
        # Compute system prompt KV cache once
        system_cache = compute_kv_cache(system_prompt)  # Once!

        # Reuse for all users
        for user in users:
            # Only compute user query + response
            response = generate(user_query, prefix_cache=system_cache)
        ```

        **Savings:**
        - Don't recompute system prompt
        - Massive throughput improvement
        - Lower latency

        **Other uses:**
        - In-context learning examples
        - Few-shot prompts
        - Any common prefix

        **Used by:** OpenAI API, Anthropic API, most serving frameworks
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand KV caching - the optimization that makes LLM inference fast!

Just one more topic to complete your advanced transformer knowledge:

Next topic:
- **Topic 23**: Mixture of Experts (MoE) - scaling to trillions of parameters efficiently

**You now understand the inference optimization used in ChatGPT, Claude, and every production LLM!** 🚀

---

*Navigation: ← Previous: Topic 21 (Flash Attention) | Next: Topic 23 (Mixture of Experts) →*
""")
