import torch
from torch import nn


class ResidualProjection(nn.Module):
    def __init__(self, in_dim, bottleneck_reduction_ratio, alpha):
        super(ResidualProjection, self).__init__()
        self.alpha = alpha
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // bottleneck_reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_dim // bottleneck_reduction_ratio, in_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x_ = self.model(x)
        return x * (1 - self.alpha) + x_ * self.alpha


def residual_adapter(model, conf):
    dim = model.model.token_embedding.weight.size()[1]
    if conf.residual_adapter.target in ['both', 'vision']:
        model.vision_adapter = ResidualProjection(dim, conf.residual_adapter.bottleneck_reduction, conf.residual_adapter.alpha)

    if conf.residual_adapter.target in ['both', 'text']:
        model.text_adapter = ResidualProjection(dim, conf.residual_adapter.bottleneck_reduction, conf.residual_adapter.alpha)

