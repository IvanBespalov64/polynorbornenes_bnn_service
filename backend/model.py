import torch
import typing

class BayesianLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters: mu and rho (sigma = log(1 + exp(rho)))
        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0)) # Small initial sigma

        # Bias parameters
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = torch.nn.Parameter(torch.Tensor(out_features).fill_(-3.0))

        # Prior variance
        self.prior_var = prior_var

    def forward(self, x, sample=True):
        if sample:
            # Sample weights using the reparameterization trick
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            w = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            b = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            w = self.weight_mu
            b = self.bias_mu
            
        return torch.nn.functional.linear(x, w, b)

    def kl_divergence(self):
        """Analytical KL divergence between q(w) and p(w) ~ N(0, prior_var)"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # KL for weights
        kl_w = 0.5 * torch.sum(
            (weight_sigma**2 + self.weight_mu**2) / self.prior_var 
            - 1 + torch.log(self.prior_var / weight_sigma**2)
        )
        # KL for bias
        kl_b = 0.5 * torch.sum(
            (bias_sigma**2 + self.bias_mu**2) / self.prior_var 
            - 1 + torch.log(self.prior_var / bias_sigma**2)
        )
        return kl_w + kl_b

def make_activation(name: str):
    name = name.lower()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(name)

class SharedBNNBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu", norm_type="layernorm"):
        super().__init__()
        self.linear = BayesianLinear(in_dim, out_dim)
        self.act = make_activation(activation)
        if norm_type == "layernorm":
            self.norm = torch.nn.LayerNorm(out_dim)
        elif norm_type == "batchnorm":
            self.norm = torch.nn.BatchNorm1d(out_dim)
        else:
            self.norm = torch.nn.Identity()

    def forward(self, x, sample=True):
        z = self.linear(x, sample)
        z = self.norm(z)
        z = self.act(z)
        return z

    def kl_total(self):
        return self.linear.kl_divergence()

class BNNHead(torch.nn.Module):
    def __init__(self, in_dim, head_hidden_dim, activation="relu"):
        super().__init__()
        self.linear_1 = BayesianLinear(in_dim, head_hidden_dim)
        self.act = make_activation(activation)
        self.linear_2 = BayesianLinear(head_hidden_dim, 1)

    def forward(self, x, sample=True):
        z = self.act(self.linear_1(x, sample))
        z = self.linear_2(z, sample)
        return z
    
    def kl_total(self):
        return self.linear_1.kl_divergence() + self.linear_2.kl_divergence()

class MTLBNNRegressor(torch.nn.Module):
    def __init__(
        self,
        task_names: typing.List[str],
        input_dim: int = 1024,
        hidden_dims: typing.Tuple[int, ...] = (512, 128),
        head_hidden_dim: int = 64,
        activation: str = "relu",
        norm_type: str = "layernorm",
        use_residual: bool = False,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.use_residual = use_residual

        blocks = []
        d_prev = input_dim
        for d in hidden_dims:
            blocks.append(SharedBNNBlock(d_prev, d, activation=activation, norm_type=norm_type))
            d_prev = d
        self.shared = torch.nn.ModuleList(blocks)
        self.shared_dim = d_prev

        self.heads = torch.nn.ModuleDict({
            t : BNNHead(self.shared_dim, head_hidden_dim, activation=activation) for t in self.task_names
        })

    def encode(self, x, sample=True):
        h = x
        for block in self.shared:
            h_new = block(h, sample)
            if self.use_residual and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        return h

    def forward(self, x, task_name: str, sample : bool = True):
        h = self.encode(x, sample)
        y = self.heads[task_name](h, sample)
        return y

    def kl_total(self):
        return torch.stack([cur.kl_total() for cur in self.shared] + [cur.kl_total() for cur in self.heads.values()]).sum()
