import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

class SurfaceEncoding(nn.Module):
    def __init__(self, in_channels: int, num_clusters: int) -> None:
        """Surface Residual Encoding. This script was influenced by DeepTEN

        Args:
            in_channels (int): number of channels to encode
            num_clusters (int): number of clusters
        """
        super(SurfaceEncoding, self).__init__()
        self.in_channels, self.num_clusters = in_channels, num_clusters
        std = 1. / ((num_clusters * in_channels)**0.5)
        self.clusters = torch.zeros(num_clusters, in_channels, dtype=torch.float, requires_grad=True).uniform_(-std, std)
        self.scale = torch.zeros(num_clusters, dtype=torch.float, requires_grad=True).uniform_(-1, 0), 

    def l2_norm(self, x: torch.Tensor, clusters: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """find weights of cluster centers using l2 distance

        Args:
            x (torch.Tensor): input features
            clusters (torch.Tensor): learned clusters
            scale (torch.Tensor): scale term

        Returns:
            torch.Tensor: cluster-wise weights
        """
        num_clusters, in_channels = clusters.size()
        batch_size = x.size(0)
        scale = scale.view((1, 1, num_clusters))
        
        x = x.unsqueeze(2).expand((batch_size, x.size(1), num_clusters, in_channels))
        clusters = clusters.view((1, 1, num_clusters, in_channels))

        norm = scale * (x - clusters).pow(2).sum(dim=3)
        return norm

    def accumelate(self, theta: torch.Tensor, x: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
        """accumelation of weighted residuals

        Args:
            theta (torch.Tensor): learned cluster weights
            x (torch.Tensor): input features
            clusters (torch.Tensor): learned clusters

        Returns:
            torch.Tensor: accumelated residuals
        """
        num_clusters, in_channels = clusters.size()
        clusters = clusters.view((1, 1, num_clusters, in_channels))
        batch_size = x.size(0)

        x = x.unsqueeze(2).expand((batch_size, x.size(1), num_clusters, in_channels))
        residual = x - clusters
        encoded_feat = (theta.unsqueeze(3)*residual).sum(dim=1)
        return encoded_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function for Surface Encoder with

        Args:
            x (torch.Tensor): input features

        Returns:
            torch.Tensor: residual features
        """
        assert x.dim() == 4 and x.size(1) == self.in_channels, x.shape
        x = Rearrange(x, 'b c h w -> b (h w) c)').contiguous()
        theta = F.softmax(self.l2_norm(x, self.clusters, self.scale), dim=2)
        # accumelate
        encoded_feat = self.accumelate(theta, x, self.clusters)
        return encoded_feat
