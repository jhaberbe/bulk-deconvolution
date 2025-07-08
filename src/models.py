import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import NegativeBinomial, Categorical, MixtureSameFamily, DirichletMultinomial

import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustMixturePrediction(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, mc_dropout=False, dropout_rate=0.2):
        super(RobustMixturePrediction, self).__init__()
        self.mc_dropout = mc_dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        x = F.relu(self.norm1(self.fc1(X)))

        # Optional dropout before residual block
        if self.mc_dropout or self.training:
            x = self.dropout(x)

        # Residual block with normalization
        residual = x
        x = F.relu(self.norm2(self.fc2(x)))
        x = x + residual

        # Optional dropout before output
        if self.mc_dropout or self.training:
            x = self.dropout(x)

        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

class MixtureToDirichlet(nn.Module):
    def __init__(self, num_components, num_features, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(num_components, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_components * num_features)
        self.num_components = num_components
        self.num_features = num_features

    def forward(self, pi):
        """
        pi: Tensor[n, C] - mixture weights per sample
        Returns:
            alpha: Tensor[n, C, F] - Dirichlet concentration parameters
        """
        x = F.relu(self.fc1(pi))                      # [n, hidden_dim]
        x = self.fc2(x).view(-1, self.num_components, self.num_features)  # [n, C, F]
        alpha = F.softplus(x) + 1e-3                  # ensure positivity
        return alpha

    @staticmethod
    def dirichlet_multinomial_loss(alpha, counts):
        """
        alpha: [n, C, F] - predicted Dirichlet params
        counts: [n, C, F] - observed counts per component per feature
        Returns:
            Negative log-likelihood (scalar)
        """
        n, C, F = counts.shape
        loss = 0.0
        for f in range(F):
            alpha_f = alpha[:, :, f]        # [n, C]
            count_f = counts[:, :, f]       # [n, C]
            total_count = count_f.sum(dim=1)  # [n]
            
            dist = DirichletMultinomial(total_count=total_count, concentration=alpha_f)
            loss -= dist.log_prob(count_f).mean()
        
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import DirichletMultinomial

class MixtureToDirichlet(nn.Module):
    def __init__(self, num_components, num_features, hidden_dim=128):
        super().__init__()
        self.num_components = num_components
        self.num_features = num_features

        # MLP will process concatenated (pi, X)
        self.fc1 = nn.Linear(num_components + num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_components * num_features)

    def forward(self, pi, X):
        """
        Args:
            pi: [n, C] - mixture weights per sample
            X:  [n, F] - observed total gene expression per sample
        Returns:
            alpha: [n, C, F] - Dirichlet parameters
        """
        x = torch.cat([pi, X], dim=1)        # [n, C + F]
        x = F.relu(self.fc1(x))              # [n, hidden_dim]
        x = self.fc2(x).view(-1, self.num_components, self.num_features)  # [n, C, F]
        alpha = F.softplus(x) + 1e-3         # ensure positivity
        return alpha

    @staticmethod
    def dirichlet_multinomial_loss(alpha, counts):
        """
        Args:
            alpha: [n, C, F] - predicted Dirichlet params
            counts: [n, C, F] - observed counts per component per feature
        Returns:
            scalar - Negative log-likelihood
        """
        n, C, F = counts.shape

        # Flatten over (n, f)
        alpha_flat = alpha.permute(0, 2, 1).reshape(-1, C)   # [n * F, C]
        counts_flat = counts.permute(0, 2, 1).reshape(-1, C) # [n * F, C]
        total_count = counts_flat.sum(dim=1)                # [n * F]

        dist = DirichletMultinomial(total_count=total_count, concentration=alpha_flat)
        return -dist.log_prob(counts_flat).mean()
