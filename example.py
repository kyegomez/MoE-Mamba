import torch 
from moe_mamba.model import MoEMamba 


# Create a tensor of shape (1, 1024, 512)
x = torch.randint(0, 10000, (1, 512))

# Create a MoEMamba model
model = MoEMamba(
    num_tokens=10000,
    dim=512,
    depth=1,
    d_state=512,
    causal=True,
    shared_qk=True,
    exact_window_size=True,
    dim_head=64,
    m_expand=4,
    num_experts=4,
)

# Forward pass
out = model(x)

# Print the shape of the output tensor
print(out)