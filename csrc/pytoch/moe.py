import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [B*S, H]

        logits = self.gate(x_flat)  # [B*S, num_experts]

        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)  # 归一化

        router_loss = self._load_balancing_loss(logits)

        return topk_weights, topk_indices, router_loss

    def _load_balancing_loss(self, logits):
        num_tokens = logits.shape[0]
        probs = F.softmax(logits, dim=-1)  # [N, E]

        _, indices = torch.topk(logits, self.top_k, dim=-1)
        one_hot = F.one_hot(indices, self.num_experts).float()  # [N, K, E]
        one_hot = one_hot.sum(dim=1)  # [N, E]
        freq = one_hot.mean(dim=0)  # [E]

        avg_prob = probs.mean(dim=0)  # [E]

        loss = self.num_experts * (freq * avg_prob).sum()
        return loss


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts=8,
        top_k=2,
        num_shared_experts=0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(hidden_dim, num_experts, top_k)

        self.experts = nn.ModuleList(
            [Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )

        self.shared_experts = nn.ModuleList(
            [Expert(hidden_dim, intermediate_dim) for _ in range(num_shared_experts)]
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [N, H], N = B*S

        topk_weights, topk_indices, router_loss = self.router(x)
        # topk_weights: [N, top_k]
        # topk_indices: [N, top_k]

        output = torch.zeros_like(x_flat)  # [N, H]

        for expert_idx in range(self.num_experts):
            mask = topk_indices == expert_idx

            if not mask.any():
                continue

            token_indices, slot_indices = torch.where(mask)
            expert_input = x_flat[token_indices]  # [num_selected, H]
            expert_output = self.experts[expert_idx](expert_input)  # [num_selected, H]
            weights = topk_weights[token_indices, slot_indices]  # [num_selected]
            output[token_indices] += weights.unsqueeze(-1) * expert_output

        for shared_expert in self.shared_experts:
            output += shared_expert(x_flat)

        output = output.view(batch_size, seq_len, hidden_dim)
        return output, router_loss


# ============================================================
# 4. Transformer Block（Attention + MoE）
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        num_experts=8,
        top_k=2,
        num_shared_experts=0,
        use_moe=True,
    ):
        super().__init__()

        # Self-Attention
        self.attn_norm = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # FFN or MoE
        self.ffn_norm = nn.RMSNorm(hidden_dim)
        self.use_moe = use_moe

        if use_moe:
            self.ffn = MoELayer(
                hidden_dim, intermediate_dim, num_experts, top_k, num_shared_experts
            )
        else:
            self.ffn = Expert(hidden_dim, intermediate_dim)

    def forward(self, x, attention_mask=None):
        router_loss = 0.0

        # --- Self-Attention + Residual ---
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = residual + x

        # --- FFN / MoE + Residual ---
        residual = x
        x = self.ffn_norm(x)
        if self.use_moe:
            x, router_loss = self.ffn(x)
        else:
            x = self.ffn(x)
        x = residual + x

        return x, router_loss


class MoEModel(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        hidden_dim=512,
        num_heads=8,
        intermediate_dim=1408,
        num_layers=12,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        moe_layer_freq=2,
        first_moe_layer=1,
        max_seq_len=2048,
        router_loss_weight=0.01,
    ):
        super().__init__()

        self.router_loss_weight = router_loss_weight

        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_moe = (i >= first_moe_layer) and (i % moe_layer_freq == 0)
            self.layers.append(
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_shared_experts=num_shared_experts if use_moe else 0,
                    use_moe=use_moe,
                )
            )

        # Output
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        causal_mask = causal_mask.to(x.device)

        # Transformer layers
        total_router_loss = 0.0
        for layer in self.layers:
            x, router_loss = layer(x, attention_mask=causal_mask)
            total_router_loss += router_loss

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, S, vocab_size]

        return logits, total_router_loss


def train_example():
    model = MoEModel(
        vocab_size=32000,
        hidden_dim=256,
        num_heads=4,
        intermediate_dim=512,
        num_layers=6,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        moe_layer_freq=2,
        first_moe_layer=1,
        max_seq_len=512,
        router_loss_weight=0.01,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    for i, layer in enumerate(model.layers):
        layer_type = "MoE" if layer.use_moe else "Dense"
        print(f"  Layer {i}: {layer_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    input_ids = torch.randint(0, 32000, (4, 128)).to(device)
    labels = torch.randint(0, 32000, (4, 128)).to(device)

    print("\n--- Training ---")
    for step in range(5):
        optimizer.zero_grad()

        logits, router_loss = model(input_ids)

        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )

        total_loss = lm_loss + model.router_loss_weight * router_loss

        total_loss.backward()
        optimizer.step()

        print(
            f"Step {step}: "
            f"total_loss={total_loss.item():.4f}, "
            f"lm_loss={lm_loss.item():.4f}, "
            f"router_loss={router_loss:.4f}"
        )

    print("\n--- Inference ---")
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, 32000, (1, 32)).to(device)
        logits, _ = model(test_input)
        predicted = logits.argmax(dim=-1)
        print(f"Input shape:  {test_input.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Predicted:    {predicted[0, :10].tolist()}")


if __name__ == "__main__":
    train_example()
