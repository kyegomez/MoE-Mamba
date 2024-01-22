[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MoE Mamba
Implementation of MoE Mamba from the paper: "MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts" in Pytorch and Zeta. 

[PAPER LINK](https://arxiv.org/abs/2401.04081)


## Install

```bash
pip install moe-mamba
```

## Usage

### `MoEMambaBlock` 
```python
import torch 
from moe_mamba import MoEMambaBlock

x = torch.randn(1, 10, 512)
model = MoEMambaBlock(
    dim=512,
    depth=6,
    d_state=128,
    expand=4,
    num_experts=4,
)
out = model(x)
print(out)

```


### `MoEMamba`

```python
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

```



## Code Quality 🧹

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)
- `black .`
- `ruff . --fix`


## Citation
```bibtex
@misc{pióro2024moemamba,
    title={MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts}, 
    author={Maciej Pióro and Kamil Ciebiera and Krystian Król and Jan Ludziejewski and Sebastian Jaszczur},
    year={2024},
    eprint={2401.04081},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

```


# License
MIT
