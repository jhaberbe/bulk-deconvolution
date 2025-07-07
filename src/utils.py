import torch 
import scanpy as sc

def scanpy_log_normalize(X: torch.Tensor, target_sum: float = 1e4):
    row_sums = X.sum(dim=1, keepdim=True).clamp(min=1e-8)
    X_norm = X / row_sums * target_sum
    return torch.log1p(X_norm)

