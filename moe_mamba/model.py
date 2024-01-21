import torch
from torch import nn
from zeta.nn import RMSNorm, MambaBlock
from swarms_torch import SwitchMoE

class MoEMambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        d_state: int,
        causal: bool = True,
        dropout: float = 0.1,
        shared_qk: bool = True,
        exact_window_size: bool = False,
        heads: int = None,
        dim_head: int = None,
        m_expand: int = 4,
        num_experts: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.causal = causal
        self.shared_qk = shared_qk
        self.exact_window_size = exact_window_size
        self.heads = heads
        self.dim_head = dim_head
        self.m_expand = m_expand
        self.num_experts = num_experts
        
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.hidden_dim = dim * m_expand

        for _ in range(depth):
            self.layers.append(
                MambaBlock(
                    dim=dim,
                    depth=depth,
                    d_state=d_state,
                    expand=m_expand,
                    *args,
                    **kwargs,
                )
            )

            self.ffn_layers.append(
                SwitchMoE(
                    dim=dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=dim,
                    num_experts=num_experts,
                    mult=m_expand,
                )
            )

    def forward(self, x):
        for attn, moe in zip(self.layers, self.ffn_layers):
            x, _ = moe(x)
            x = attn(x, x, x) + x
            x, _ = moe(x)
        return x

x = torch.randn(1, 10, 512)
model = MoEMambaBlock(
    dim=512,
    depth=6,
    d_state=128,
    expand=4,
    num_experts=4,
)
model(x).shape










# torch.Size([1, 10, 512])

# classes


# class MoEMamba(nn.Module):
#     def __init__(
#         self,
#         num_tokens: int,
#         dim: int,
#         depth: int,
#         d_state: int,
#         causal: bool = True,
#         dropout: float = 0.1,
#         shared_qk: bool = True,
#         exact_window_size: bool = False,
#         heads: int = None,
#         dim_head: int = None,
#         m_expand: int = 4,
#         num_experts: int = 4,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.emb = nn.Embedding(num_tokens, dim)

#         self.transformer = MoEMambaBlock(
#             dim, depth, heads, dim_head, ff_mult
#         )

#         self.to_logits = nn.Sequential(
#             RMSNorm(dim), nn.Linear(dim, num_tokens)
#         )

#     def forward(self, x):
#         x = self.emb(x)
#         x = self.transformer(x)
#         return self.to_logits(x)
