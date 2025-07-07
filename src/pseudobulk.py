import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm, trange

class PseudoBulkGenerator:
    # pbulk = PseudoBulkGenerator(adata, min_cells=1000, max_cells=3000, alpha=0.5, random_state=42).generate(n_iterations=10_000)

    def __init__(self, adata, min_cells=1000, max_cells=3000, alpha=0.5, random_state=None):
        """
        Initialize pseudobulk generator with Dirichlet sampling.
        
        Parameters:
            adata: AnnData object with 'cell_type' in obs and counts in layers["counts"]
            min_cells, max_cells: uniform range of total cells per pseudobulk
            alpha: Dirichlet concentration parameter; lower=more variance
            random_state: optional int for reproducibility
        """
        self.adata = adata
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.alpha = alpha

        if random_state is not None:
            np.random.seed(random_state)
        
        # Get unique classes and validate
        self.labels = adata.obs["cell_type"].to_numpy()
        self.unique_classes = np.unique(self.labels)
        self.n_classes = len(self.unique_classes)

    def generate(self, n_iterations=1000):
        counts = []
        cell_type_summaries = []
        per_label_contributions = {}

        for sample_id in trange(n_iterations, desc="Generating pseudobulk samples"):
            # 1) Dirichlet sample: random but balanced probabilities
            dirichlet_probs = np.random.dirichlet(self.alpha * np.ones(self.n_classes))

            # 2) Random total number of cells to sample
            total_cells = np.random.randint(self.min_cells, self.max_cells + 1)

            # 3) Determine number of cells per class with a multinomial
            samples_per_class = np.random.multinomial(total_cells, dirichlet_probs)

            # 4) Collect sampled indices and class summary
            sampled_indices = []
            class_counts = {}

            cell_type_counts = {}
            for class_label, n_samples in zip(self.unique_classes, samples_per_class):
                # Filter adata for this class
                class_mask = self.labels == class_label
                class_indices = np.where(class_mask)[0]

                # Sample with replacement from this class
                chosen = np.random.choice(class_indices, size=n_samples, replace=True)
                cell_type_counts[class_label] = np.array(self.adata.layers["counts"][chosen].sum(axis=0)).ravel()
                sampled_indices.append(chosen)

                # Record how many of this class were sampled
                class_counts[class_label] = n_samples
            
            per_label_contributions[sample_id] = cell_type_counts

            sampled_indices = np.concatenate(sampled_indices)
            np.random.shuffle(sampled_indices)

            # Sum counts for this pseudobulk
            bulk_counts = np.array(self.adata.layers["counts"][sampled_indices].sum(axis=0)).ravel()
            counts.append(bulk_counts)
            cell_type_summaries.append(class_counts)

        adata = ad.AnnData(
            X=np.array(counts),
            obs=pd.DataFrame(cell_type_summaries).fillna(0).astype(int),
            var=pd.DataFrame(index=self.adata.var.index)
        )
        
        adata.uns["per_label_contributions"] = per_label_contributions

        for key in tqdm(adata.uns["per_label_contributions"][0]):
            adata.layers[key] = np.stack([adata.uns["per_label_contributions"][i][key] for i in adata.uns["per_label_contributions"]], axis=0)
        del adata.uns

        return adata