import torch
from torch import nn
from zeta.nn import RMSNorm, MambaBlock
from swarms_torch import SwitchMoE

class MoEMambaBlock(nn.Module):
    """
    MoEMambaBlock is a module that combines MambaBlock and SwitchMoE layers.
    
    Args:
        dim (int): The input dimension.
        depth (int): The number of MambaBlock layers.
        d_state (int): The dimension of the state.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        shared_qk (bool, optional): Whether to share the query and key projections. Defaults to True.
        exact_window_size (bool, optional): Whether to use exact window size for attention. Defaults to False.
        heads (int, optional): The number of attention heads. Defaults to None.
        dim_head (int, optional): The dimension of each attention head. Defaults to None.
        m_expand (int, optional): The expansion factor for the hidden dimension. Defaults to 4.
        num_experts (int, optional): The number of experts in the SwitchMoE layer. Defaults to 4.
    """
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
                )
            )

    def forward(self, x):
        """
        Forward pass of the MoEMambaBlock module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        for mamba, moe in zip(self.layers, self.ffn_layers):
            x = mamba(x)
            x, _ = moe(x)
        return x












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
