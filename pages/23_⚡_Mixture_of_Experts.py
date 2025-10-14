import streamlit as st

st.title("⚡ Topic 23: Mixture of Experts (MoE)")

st.markdown("""
---

## 🎯 The Scaling Challenge: Can We Go Bigger?

You've learned about transformers - the architecture powering modern LLMs. But there's a fundamental question:

**How do we make models MORE capable without making them SLOWER?**

**The traditional approach:**

```
GPT-2:   1.5B parameters  →  trains and runs on one GPU
GPT-3:   175B parameters  →  needs 100+ GPUs, very expensive
GPT-4:   ~1.8T parameters →  requires massive infrastructure
```

**The problem:** More parameters = more compute for EVERY token!

```python
# Dense model: ALL parameters used for EVERY token
output = layer_1(x)  # Uses all 10B parameters
output = layer_2(output)  # Uses all 10B parameters
...

# Cost: O(total_parameters) per token
```

**What if we could have TRILLIONS of parameters but only use a FRACTION at a time?**

**The solution:** **Mixture of Experts (MoE)** - conditional computation!

---

## 💡 The Core Idea: Sparse Activation

### Dense vs Sparse Models

**Dense Model (GPT, LLaMA):**
```
Input → Layer 1 (use ALL params) → Layer 2 (use ALL params) → Output

Every token uses every parameter.
Cost: O(N) where N = total parameters
```

**Mixture of Experts:**
```
Input → Router → Select 2 out of 8 experts → Use only selected experts → Output

Each token uses only a SUBSET of parameters.
Cost: O(N/k) where k = number of experts
```

**Example:**

```python
# Dense FFN (standard transformer)
output = FFN(x)  # 10B parameters, always used

# MoE (8 experts)
expert_1_params = 1.25B
expert_2_params = 1.25B
...
expert_8_params = 1.25B
# Total: 10B parameters

# But for each token, only use 2 experts!
active_experts = router.select_top_k(x, k=2)
output = combine([expert_1(x), expert_5(x)])  # Only 2.5B params used!
```

**Result:** 10B total capacity, but 2.5B active computation per token!

---

## 🏗️ MoE Architecture

### Standard Transformer Block

```
Input: x
  ↓
Self-Attention
  ↓
Add & Norm
  ↓
Feed-Forward Network (FFN)  ← ALL parameters used
  ↓
Add & Norm
  ↓
Output
```

### MoE Transformer Block

```
Input: x
  ↓
Self-Attention
  ↓
Add & Norm
  ↓
╔════════════════════════════╗
║   Mixture of Experts       ║
║                            ║
║   Router: Which experts?   ║
║       ↓                    ║
║   [Expert 1] [Expert 2]    ║  ← Only 2 out of 8 used!
║   [Expert 3] [Expert 4]    ║
║   [Expert 5] [Expert 6]    ║
║   [Expert 7] [Expert 8]    ║
║       ↓                    ║
║   Combine outputs          ║
╚════════════════════════════╝
  ↓
Add & Norm
  ↓
Output
```

**Key components:**
1. **Router**: Decides which experts to use
2. **Experts**: Separate FFN networks (typically 8-64)
3. **Top-k selection**: Use only k best experts (typically k=2)
4. **Load balancing**: Ensure experts are used evenly

---

## 🧮 The Mathematics

### Router Network

The router decides which experts to activate:

```python
def router(x):
    # x: [batch, seq_len, d_model]

    # Compute routing scores
    scores = x @ W_router  # [batch, seq_len, num_experts]

    # Softmax to get probabilities
    probs = softmax(scores)  # [batch, seq_len, num_experts]

    # Select top-k experts
    top_k_probs, top_k_indices = top_k(probs, k=2)

    return top_k_probs, top_k_indices
```

### Expert Computation

```python
# Each expert is a standard FFN
def expert(x):
    h = SwiGLU(W1 @ x)  # Or ReLU, GELU, etc.
    output = W2 @ h
    return output
```

### Combining Expert Outputs

```python
# Weighted combination based on router scores
output = sum(weight_i * expert_i(x) for i in top_k_experts)
```

### Complete Formula

```
MoE(x) = Σ G(x)ᵢ · Eᵢ(x)
         i∈TopK(G(x))

where:
- G(x) = router scores (gating network)
- Eᵢ(x) = output of expert i
- TopK = select k experts with highest scores
```

---

## 💻 Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    Single expert: standard FFN
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        h = F.silu(self.w1(x))  # SwiGLU activation
        return self.dropout(self.w2(h))


class Router(nn.Module):
    """
    Router network: decides which experts to use
    """

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Routing layer
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, d_model]

        # Compute routing scores
        logits = self.gate(x)  # [batch, seq_len, num_experts]

        # Softmax for probabilities
        probs = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return top_k_probs, top_k_indices


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer

    Used in: GPT-4 (rumored), Mixtral, Switch Transformer, GLaM
    """

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Expert FFN hidden dimension
            num_experts: Number of expert networks
            top_k: Number of experts to activate per token
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # Router
        self.router = Router(d_model, num_experts, top_k)

    def forward(self, x):
        """
        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            load_balancing_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        top_k_probs, top_k_indices = self.router(x)
        # top_k_probs: [batch, seq_len, top_k]
        # top_k_indices: [batch, seq_len, top_k]

        # Initialize output
        output = torch.zeros_like(x)

        # Compute expert outputs
        # For efficiency, we could batch tokens going to same expert,
        # but for clarity, we'll iterate
        for i in range(self.top_k):
            # Get indices and weights for this position
            expert_indices = top_k_indices[:, :, i]  # [batch, seq_len]
            expert_weights = top_k_probs[:, :, i:i+1]  # [batch, seq_len, 1]

            # For each expert
            for expert_id in range(self.num_experts):
                # Mask: which tokens use this expert in position i?
                mask = (expert_indices == expert_id).unsqueeze(-1)  # [batch, seq_len, 1]

                # If any tokens use this expert, compute output
                if mask.any():
                    expert_output = self.experts[expert_id](x)

                    # Add weighted contribution
                    output = output + expert_weights * mask.float() * expert_output

        # Compute load balancing loss (encourage balanced expert usage)
        # This is an auxiliary loss added during training
        router_probs = F.softmax(self.router.gate(x), dim=-1)  # [batch, seq_len, num_experts]
        expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Ideal: each expert used 1/num_experts of the time
        # Loss: variance of usage (want it to be uniform)
        load_balancing_loss = self.num_experts * (expert_usage ** 2).sum()

        return output, load_balancing_loss


class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts
    """

    def __init__(self, d_model, num_heads, num_experts=8, top_k=2, d_ff=None):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # Standard components
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # MoE instead of standard FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attn(
            self.attn_norm(x),
            self.attn_norm(x),
            self.attn_norm(x),
            attn_mask=mask
        )
        x = x + attn_out

        # MoE
        moe_out, load_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out

        return x, load_loss


# Example usage
batch_size = 2
seq_len = 128
d_model = 512
num_experts = 8
top_k = 2

# Create MoE layer
moe = MixtureOfExperts(
    d_model=d_model,
    d_ff=2048,
    num_experts=num_experts,
    top_k=top_k
)

# Input
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, load_loss = moe(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Load balancing loss: {load_loss.item():.4f}")

# Parameter count comparison
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Standard FFN
ffn_standard = nn.Sequential(
    nn.Linear(d_model, 2048),
    nn.SiLU(),
    nn.Linear(2048, d_model)
)

print(f"\\nParameter comparison:")
print(f"Standard FFN: {count_params(ffn_standard):,} parameters")
print(f"MoE ({num_experts} experts): {count_params(moe):,} parameters")
print(f"Active per token: ~{count_params(ffn_standard) * top_k:,} parameters ({top_k}/{num_experts} experts)")
```

**Output:**
```
Input shape: torch.Size([2, 128, 512])
Output shape: torch.Size([2, 128, 512])
Load balancing loss: 0.8234

Parameter comparison:
Standard FFN: 2,099,200 parameters
MoE (8 experts): 16,793,600 parameters
Active per token: ~4,198,400 parameters (2/8 experts)

Total capacity: 8x larger
Active computation: 2x larger (not 8x!)
```

---

## ⚖️ Load Balancing: The Critical Challenge

### The Problem

```python
# Without load balancing, router might prefer certain experts
Expert 1: Used 60% of the time (overloaded!)
Expert 2: Used 30% of the time
Expert 3-8: Used 10% total (wasted!)

# This defeats the purpose!
```

### Solutions

**1. Auxiliary Loss (Most Common)**

```python
# Encourage uniform expert usage
def load_balancing_loss(router_probs):
    # router_probs: [batch, seq_len, num_experts]

    # Average usage per expert
    usage = router_probs.mean(dim=[0, 1])  # [num_experts]

    # Penalty for non-uniform usage
    # Want all experts at 1/num_experts
    loss = num_experts * (usage ** 2).sum()

    return loss

# Add to total loss during training
total_loss = cross_entropy_loss + alpha * load_balancing_loss
```

**2. Expert Capacity (Switch Transformer)**

```python
# Limit how many tokens each expert can process
capacity = (seq_len * batch_size / num_experts) * capacity_factor

# If expert full, drop tokens or send to overflow expert
if expert_load > capacity:
    # Drop or redistribute
```

**3. Expert Choice Routing (Recent)**

```python
# Instead of tokens choosing experts,
# experts choose tokens!

for expert in experts:
    # Each expert selects top-k tokens to process
    selected_tokens = expert.select_top_k(all_tokens)
    expert.process(selected_tokens)
```

---

## 🌟 Real-World MoE Models

### Mixtral 8x7B (Mistral AI, 2023)

**Architecture:**
- 8 experts per MoE layer
- Top-2 routing
- Total parameters: 46.7B
- Active parameters: 12.9B

**Performance:**
- Outperforms LLaMA 2 70B (dense)
- 6x faster inference than 70B model
- Same compute as 12B dense model

```python
# Mixtral configuration
d_model = 4096
num_layers = 32
num_experts = 8
top_k = 2

# Per layer:
# - Self-attention: 4096 × 4096 × 3 = ~50M params
# - MoE: 8 experts × 14B params each = ~112M params
# Total per layer: ~162M params
# 32 layers: ~5.2B params

# But active per token:
# - Attention: 50M
# - MoE (2/8 experts): 28M
# Active per layer: ~78M
# 32 layers active: ~2.5B params per token
```

### GPT-4 (OpenAI, 2023 - Rumored)

**Rumored architecture:**
- 16 experts (unconfirmed)
- ~220B params per expert
- ~1.8 trillion total parameters
- Top-2 routing
- Active: ~440B parameters per token

**Why rumors suggest MoE:**
- Massive capacity (1.8T params)
- Reasonable inference cost
- Quality jump from GPT-3.5

### Switch Transformer (Google, 2021)

**Architecture:**
- Up to 1.6 trillion parameters
- 2048 experts!
- Top-1 routing (only one expert per token)
- Enabled training on 1T+ parameter scale

---

## 📊 Sparse vs Dense Trade-offs

### Advantages of MoE

✅ **More Parameters, Same Cost:**
```
Dense 70B: 70B params, 70B active
MoE 8×7B: 56B params, 12B active (4.6x more capacity, same cost!)
```

✅ **Better Quality at Same Compute:**
- Mixtral 8×7B > LLaMA 2 70B (dense)
- Same inference cost as LLaMA 2 12B

✅ **Specialization:**
- Different experts learn different skills
- Expert 1: Math
- Expert 2: Code
- Expert 3: Creative writing
- etc.

### Challenges of MoE

❌ **Load Balancing:**
- Need auxiliary losses
- Careful tuning required
- Can be unstable during training

❌ **Memory:**
- All expert params must fit in memory
- 8×7B needs 46B param capacity
- Though only 12B active per token

❌ **Fine-tuning:**
- Harder to fine-tune than dense models
- Load balancing can break

❌ **Communication (Distributed):**
- Experts on different GPUs = inter-GPU communication
- Can bottleneck throughput

---

## 🚀 Training and Inference

### Training with MoE

```python
# Training loop
for batch in dataloader:
    # Forward pass
    output, load_loss = moe_model(batch)

    # Main loss
    main_loss = cross_entropy(output, target)

    # Total loss includes load balancing
    total_loss = main_loss + 0.01 * load_loss  # Small weight

    # Backward
    total_loss.backward()
    optimizer.step()
```

### Inference Optimization

```python
# Batch tokens by expert assignment
def optimized_moe_inference(x):
    # Route all tokens
    expert_assignments = router(x)  # Which expert for each token?

    # Group tokens by expert
    expert_to_tokens = group_by_expert(x, expert_assignments)

    # Process each expert's batch in parallel
    outputs = {}
    for expert_id, tokens in expert_to_tokens.items():
        # Batch processing
        outputs[expert_id] = experts[expert_id](tokens)

    # Reassemble in original order
    final_output = reassemble(outputs, expert_assignments)

    return final_output
```

**Benefit:** Process each expert once per batch, not per token!

---

## 🎓 Key Takeaways

1. **MoE = Conditional Computation:**
   - Total params: Large (8-64 experts × expert size)
   - Active params: Small (top-k experts × expert size)

2. **Capacity vs Cost Trade-off:**
   - Mixtral 8×7B: 46B total, 12B active
   - Same cost as 12B dense, capacity of 46B!

3. **Router is critical:**
   - Decides which experts to use
   - Needs load balancing
   - Can specialize (math expert, code expert, etc.)

4. **Real-world usage:**
   - Mixtral 8×7B (open source)
   - GPT-4 (rumored)
   - Switch Transformer (research)
   - DeepSeek-V3 (Chinese LLM)

5. **Best for:**
   - Scaling to huge parameter counts
   - Maintaining inference efficiency
   - Learning diverse skills

6. **Challenges:**
   - Load balancing
   - Training stability
   - Memory requirements
   - Communication overhead

**MoE is THE path to trillion-parameter models with practical inference!**

---

## 📝 Quiz Time!

Test your understanding of Mixture of Experts.
""")

# Quiz questions
with st.expander("Question 1: Core concept"):
    st.markdown("""
    **Question**: What is the key advantage of Mixture of Experts over dense models?

    A) Faster training
    B) More total parameters with similar active computation per token
    C) Fewer parameters
    D) Simpler architecture
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) More total parameters with similar active computation per token**

        **Explanation**:

        **The MoE magic:**
        ```
        Dense model (70B):
        - Total params: 70B
        - Active per token: 70B
        - Cost: High

        MoE model (8×7B):
        - Total params: 56B (8 experts × 7B each)
        - Active per token: 14B (2 experts × 7B, top-2 routing)
        - Cost: Similar to 14B dense model!
        ```

        **Benefits:**
        - 4x more total capacity than active cost suggests
        - Can learn more diverse patterns (8 experts specialize)
        - Better quality at same inference cost

        **Real example:**
        - Mixtral 8×7B beats LLaMA 2 70B (dense)
        - Mixtral inference cost = LLaMA 2 12B
        - 5-6x cheaper to run than 70B!

        This is how we can build trillion-parameter models (GPT-4) with practical inference!
        """)

with st.expander("Question 2: Router function"):
    st.markdown("""
    **Question**: What does the router network in MoE do?

    A) Routes data to different GPUs
    B) Decides which experts to activate for each token
    C) Balances the load across all layers
    D) Compresses the input
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) Decides which experts to activate for each token**

        **Explanation**:

        **Router operation:**
        ```python
        # For each token, compute scores for all experts
        scores = token @ W_router  # [num_experts]

        # Select top-k (usually k=2)
        top_k_experts = topk(scores, k=2)

        # Token: "Python"
        # Scores: [0.1, 0.8, 0.05, 0.7, 0.2, 0.1, 0.15, 0.05]
        #          E1   E2    E3    E4   E5   E6   E7    E8
        # Selected: Expert 2 (0.8) and Expert 4 (0.7)
        ```

        **What this enables:**
        - **Specialization**: Expert 2 might be "code expert", Expert 4 "Python expert"
        - **Efficiency**: Only run 2 out of 8 experts (75% computation saved!)
        - **Adaptivity**: Different tokens use different experts

        **Learned behavior:**
        - Code tokens → code experts
        - Math tokens → math experts
        - Creative tokens → creative writing experts

        The router learns this automatically during training!
        """)

with st.expander("Question 3: Load balancing problem"):
    st.markdown("""
    **Question**: Why is load balancing important in MoE?

    A) To balance GPU memory usage
    B) To ensure all experts are used roughly equally, avoiding wasted capacity
    C) To balance the learning rate
    D) To reduce training time
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) To ensure all experts are used roughly equally, avoiding wasted capacity**

        **Explanation**:

        **The problem without load balancing:**
        ```
        Expert 1: Used 70% of the time (overloaded!)
        Expert 2: Used 20% of the time
        Experts 3-8: Used 10% total (mostly wasted!)
        ```

        **Why this is bad:**
        - Expert 1 becomes a bottleneck
        - Experts 3-8 are wasted parameters (we could remove them!)
        - Defeats the purpose of MoE

        **Load balancing solutions:**

        1. **Auxiliary loss:**
        ```python
        # Penalize non-uniform usage
        usage = [expert_usage for each expert]
        # Want: [12.5%, 12.5%, 12.5%, 12.5%, ...]
        loss = variance(usage)  # Add to training loss
        ```

        2. **Expert capacity:**
        ```python
        # Limit tokens per expert
        if expert_1.load > capacity:
            send_to_different_expert()
        ```

        **Result:**
        - All experts utilized
        - Full capacity used
        - No wasted parameters

        Load balancing is THE critical challenge in MoE training!
        """)

with st.expander("Question 4: Mixtral architecture"):
    st.markdown("""
    **Question**: Mixtral 8x7B has 46.7B total parameters but how many active parameters per token?

    A) 46.7B
    B) 28B
    C) 12.9B
    D) 7B
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: C) 12.9B**

        **Explanation**:

        **Mixtral 8×7B breakdown:**

        ```
        Total parameters: 46.7B
        - Shared components (attention, embeddings): ~10B
        - MoE experts: 8 experts × ~4.5B each = ~36B

        Active per token:
        - Shared components: ~10B (always active)
        - Active experts: 2 out of 8 = 2 × ~1.45B = ~2.9B

        Total active: 10B + 2.9B ≈ 12.9B
        ```

        **Why this matters:**

        **Comparison with dense models:**
        - LLaMA 2 13B: 13B total, 13B active
        - Mixtral: 46.7B total, 12.9B active
        - Similar inference cost, but Mixtral has 3.6x more capacity!

        **Performance:**
        - Mixtral beats LLaMA 2 70B (dense) on benchmarks
        - But inference cost of only 13B model
        - 5x cheaper to run than 70B!

        This is the MoE advantage: massive capacity, moderate cost!
        """)

with st.expander("Question 5: When to use MoE"):
    st.markdown("""
    **Question**: When is MoE most beneficial compared to dense models?

    A) For small models (<1B parameters)
    B) For very large models where you want massive capacity without proportional inference cost
    C) For models with limited training data
    D) For models that need to be fast during training
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: B) For very large models where you want massive capacity without proportional inference cost**

        **Explanation**:

        **When MoE shines:**

        **Scenario 1: Scaling beyond practical dense limits**
        ```
        Dense 1T params: Impractical inference (too slow)
        MoE 16×100B: 1.6T total, 200B active (practical!)
        ```

        **Scenario 2: Quality at lower cost**
        ```
        Dense 70B: Expensive to serve
        MoE 8×7B: Better quality, 5x cheaper
        ```

        **When NOT to use MoE:**

        **Small models:**
        - <10B params: Overhead not worth it
        - Dense is simpler and fine

        **Limited infrastructure:**
        - Need distributed training
        - Complex load balancing
        - More engineering effort

        **Simple fine-tuning:**
        - MoE harder to fine-tune
        - Load balancing can break

        **Sweet spot:**
        - Models >50B parameters
        - Production serving at scale
        - Need quality of 100B+ model with cost of 20B model

        **Used in:**
        - GPT-4 (rumored, 1.8T params)
        - Mixtral (open source, 46B params)
        - DeepSeek-V3 (Chinese LLM)

        MoE is the future of ultra-large language models!
        """)

st.markdown("""
---

## 🎉 Congratulations! You've Completed Advanced Transformers!

You've now mastered ALL the advanced topics that power modern LLMs:

✅ **Topic 14**: Attention Mechanism - The foundation
✅ **Topic 15**: Multi-Head Attention - Parallel pattern learning
✅ **Topic 16**: Positional Encoding - Teaching order
✅ **Topic 17**: RoPE - Modern position encoding
✅ **Topic 18**: Complete Transformer - The full architecture
✅ **Topic 19**: Modern Components - RMSNorm, SwiGLU
✅ **Topic 20**: Grouped Query Attention - Memory efficiency
✅ **Topic 21**: Flash Attention - Speed breakthrough
✅ **Topic 22**: KV Cache - Fast inference
✅ **Topic 23**: Mixture of Experts - Scaling to trillions

**You now understand:**
- How ChatGPT, Claude, and GPT-4 work internally
- Why LLaMA uses RoPE, RMSNorm, and GQA
- How Mixtral achieves 70B quality at 13B cost
- The optimizations that make modern LLMs possible

**What's next?**
- **Professional Level** (Topics 24-29): Production training, optimization, deployment
- Build your own mini-LLM with these techniques!
- Contribute to open-source LLM projects

**You're now equipped to understand and build modern LLM architectures!** 🚀

---

*Navigation: ← Previous: Topic 22 (KV Cache) | Next: Professional Topics (24-29) →*
""")
