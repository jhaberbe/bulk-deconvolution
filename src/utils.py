import torch 
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse



def scanpy_log_normalize(X: torch.Tensor, target_sum: float = 1e4):
    row_sums = X.sum(dim=1, keepdim=True).clamp(min=1e-8)
    X_norm = X / row_sums * target_sum
    return torch.log1p(X_norm)

def collapse_duplicate_var(adata: ad.AnnData) -> ad.AnnData:
    """
    Collapse duplicate var.index entries in an AnnData object by summing across columns (features),
    in both `X` and all matrices in `layers`. Returns a new AnnData object with unique var.index.
    """
    var_idx = adata.var.index

    # Get unique gene names and mapping
    grouped = pd.Series(np.arange(len(var_idx)), index=var_idx)
    unique_genes = grouped.groupby(level=0).apply(list)

    # Initialize containers for new matrices
    def collapse_matrix(mat):
        if sparse.issparse(mat):
            collapsed = sparse.hstack([
                mat[:, idxs].sum(axis=1) for idxs in unique_genes
            ]).tocsr()
        else:
            collapsed = np.stack([
                mat[:, idxs].sum(axis=1) for idxs in unique_genes
            ], axis=1)
        return collapsed

    # Collapse X
    X_collapsed = collapse_matrix(adata.X)

    # Collapse layers
    layers_collapsed = {}
    for key, mat in adata.layers.items():
        layers_collapsed[key] = collapse_matrix(mat)

    # Create new var dataframe
    var_new = adata.var.groupby(var_idx).first().loc[unique_genes.index]
    var_new.index.name = None  # remove name if present

    # Create new AnnData
    new_adata = ad.AnnData(
        X=X_collapsed,
        obs=adata.obs.copy(),
        var=var_new,
        layers=layers_collapsed,
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        varm=adata.varm.copy(),
        obsp=adata.obsp.copy(),
        varp=adata.varp.copy()
    )

    return new_adata
