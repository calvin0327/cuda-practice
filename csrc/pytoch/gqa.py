import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=bias)

    def forward(self, x, mask=None):
        B, S, _ = x.shape  # [B, S, D]

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn = (q @ k.transpose(-1, -2)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, Hq, S, d]

        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        return self.out_proj(out)


def test_gqa():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2  # batch size
    S = 128  # sequence length
    D = 4096  # hidden size
    Hq = 32  # query heads
    Hkv = 8  # kv heads (GQA ratio = 4:1)

    print("=" * 60)
    print("GQA 测试")
    print(f"Config: B={B}, S={S}, D={D}, Hq={Hq}, Hkv={Hkv}")
    print(f"Group size: {Hq // Hkv} (每 {Hq // Hkv} 个 Q 头共享 1 组 KV)")
    print("=" * 60)

    # 创建模型
    gqa = GQA(
        hidden_size=D,
        q_head_nums=Hq,
        kv_head_nums=Hkv,
        bias=False,
    ).to(device)

    x = torch.randn(B, S, D, device=device)

    output, weights = gqa(x)

    print(f"\n--- Shape 验证 ---")
    print(f"Input:          {x.shape}")  # (2, 128, 4096)
    print(f"Output:         {output.shape}")  # (2, 128, 4096)
    print(f"Attn Weights:   {weights.shape}")  # (2, 32, 128, 128)

    assert output.shape == (B, S, D), f"输出 shape 错误: {output.shape}"
    assert weights.shape == (B, Hq, S, S), f"权重 shape 错误: {weights.shape}"
    print("✅ Shape 验证通过!")


if __name__ == "__main__":
    test_gqa()
