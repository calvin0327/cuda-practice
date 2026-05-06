import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_nums = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor):
        B, S, D = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.head_nums, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.head_nums, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.head_nums, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


if __name__ == "__main__":
    batch_size = 2
    seq_lens = 128
    hidden_size = 4096

    mha = MHA(hidden_size=hidden_size, num_heads=32)
    x = torch.randn(batch_size, seq_lens, hidden_size)
    output, weights = mha(x)

    print("output shape", output.shape)
    print("weights shape", weights.shape)
