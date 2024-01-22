import torch
import torch.nn.functional as F
from swarms_torch import SwitchMoE
from torch import Tensor, nn
from zeta.nn import FeedForward, MambaBlock, RMSNorm


class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(1, top_k_indices, 1)

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim, mult, *args, **kwargs)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(x, use_aux_loss=self.use_aux_loss)

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss


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









class MoEMamba(nn.Module):
    """
    MoEMamba is a PyTorch module that implements the MoE-Mamba model.

    Args:
        num_tokens (int): The number of tokens in the input vocabulary.
        dim (int): The dimension of the token embeddings.
        depth (int): The depth of the MoE-Mamba model.
        d_state (int): The dimension of the state in the MoE-Mamba model.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        shared_qk (bool, optional): Whether to share the query and key projections. Defaults to True.
        exact_window_size (bool, optional): Whether to use exact window size for local attention. Defaults to False.
        heads (int, optional): The number of attention heads. If None, it is set to `dim // dim_head`. Defaults to None.
        dim_head (int, optional): The dimension of each attention head. Defaults to None.
        m_expand (int, optional): The expansion factor for the MoE-Mamba model. Defaults to 4.
        num_experts (int, optional): The number of experts in the MoE-Mamba model. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        emb (nn.Embedding): The embedding layer for token embeddings.
        transformer (MoEMambaBlock): The MoE-Mamba block.
        to_logits (nn.Sequential): The sequential layer for converting the output to logits.

    """

    def __init__(
        self,
        num_tokens: int,
        dim,
        depth,
        d_state: int,
        causal: bool = True,
        dropout: float = 0.1,
        shared_qk: bool = True,
        exact_window_size: bool = False,
        dim_head: int = 64,
        m_expand: int = 4,
        num_experts: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.mamba_block = MoEMambaBlock(
            dim=dim,
            depth=depth,
            d_state=d_state,
            causal=causal,
            dropout=dropout,
            shared_qk=shared_qk,
            exact_window_size=exact_window_size,
            dim_head=dim_head,
            m_expand=m_expand,
            num_experts=num_experts,
            *args,
            **kwargs,
        )

        self.to_logits = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        """
        Forward pass of the MoEMamba model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, num_tokens).

        """
        x = self.emb(x)
        print(x.shape)
        x = self.mamba_block(x)
        return self.to_logits(x)
