import torch
import torch.nn as nn
import torch.nn.functional as F

from .wsi_model import RRTMIL


class DenseNet121FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        try:
            from monai.networks.nets import DenseNet121
        except ImportError as exc:
            raise ImportError("IADA MRI encoder requires MONAI: pip install monai") from exc
        self.backbone = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2)
        self.out_dim = 1024

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.class_layers.relu(x)
        x = self.backbone.class_layers.pool(x)
        return torch.flatten(x, 1)


class DistributionHead(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.sigma = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + 1e-6
        logvar = 2.0 * torch.log(sigma)
        return mu, sigma, logvar


class FeatureGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


def gaussian_kl(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl = 0.5 * (
        logvar2
        - logvar1
        + (var1 + (mu1 - mu2).pow(2)) / (var2 + 1e-8)
        - 1.0
    )
    return kl.sum(dim=1).mean()


def standard_normal_kl(mu, logvar):
    var = torch.exp(logvar)
    kl = 0.5 * (mu.pow(2) + var - 1.0 - logvar)
    return kl.sum(dim=1).mean()


@torch.no_grad()
def dgm_coefficients(wsi_logits, mri_logits, labels, beta=0.1):
    labels = labels.long().view(-1)
    prob_wsi = F.softmax(wsi_logits, dim=1)
    prob_mri = F.softmax(mri_logits, dim=1)
    score_wsi = prob_wsi.gather(1, labels.unsqueeze(1)).sum()
    score_mri = prob_mri.gather(1, labels.unsqueeze(1)).sum()

    ratio_wsi = score_wsi / (score_mri + 1e-8)
    ratio_mri = 1.0 / (ratio_wsi + 1e-8)

    coeff_wsi = torch.ones_like(ratio_wsi)
    coeff_mri = torch.ones_like(ratio_mri)
    if ratio_wsi > 1:
        coeff_wsi = 1.0 - torch.tanh(beta * ratio_wsi)
    else:
        coeff_mri = 1.0 - torch.tanh(beta * ratio_mri)

    return {
        "ratio_wsi": float(ratio_wsi.detach().cpu()),
        "ratio_mri": float(ratio_mri.detach().cpu()),
        "coeff_wsi": float(coeff_wsi.detach().cpu()),
        "coeff_mri": float(coeff_mri.detach().cpu()),
    }


def apply_dgm(model, coeffs):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if model.is_wsi_parameter(name):
            param.grad.mul_(coeffs["coeff_wsi"])
        elif model.is_mri_parameter(name):
            param.grad.mul_(coeffs["coeff_mri"])


class IADA(nn.Module):
    def __init__(
        self,
        n_classes=2,
        wsi_input_dim=1024,
        wsi_embed_dim=512,
        mri_in_channels=1,
        hidden_dim=256,
        latent_dim=256,
        dropout=0.25,
        rrt_region_num=8,
        rrt_layers=2,
        rrt_heads=8,
        rrt_epeg=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.wsi_encoder = RRTMIL(
            input_dim=wsi_input_dim,
            mlp_dim=wsi_embed_dim,
            n_classes=n_classes,
            dropout=dropout,
            region_num=rrt_region_num,
            n_layers=rrt_layers,
            n_heads=rrt_heads,
            epeg=rrt_epeg,
            fc=False,
        )
        self.mri_encoder = DenseNet121FeatureExtractor(in_channels=mri_in_channels)

        self.wsi_projector = nn.Sequential(
            nn.Linear(wsi_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mri_projector = nn.Sequential(
            nn.Linear(self.mri_encoder.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.wsi_distribution = DistributionHead(hidden_dim, latent_dim)
        self.mri_distribution = DistributionHead(hidden_dim, latent_dim)
        self.wsi_gate = FeatureGate(latent_dim)
        self.mri_gate = FeatureGate(latent_dim)

        self.fusion_classifier = nn.Linear(latent_dim * 2, n_classes)

    @staticmethod
    def is_wsi_parameter(name):
        return name.startswith(("wsi_encoder", "wsi_projector", "wsi_distribution", "wsi_gate"))

    @staticmethod
    def is_mri_parameter(name):
        return name.startswith(("mri_encoder", "mri_projector", "mri_distribution", "mri_gate"))

    def encode_wsi(self, wsi):
        if isinstance(wsi, (list, tuple)):
            features = [self.wsi_encoder(item) for item in wsi]
            return torch.cat([f if f.ndim == 2 else f.unsqueeze(0) for f in features], dim=0)
        return self.wsi_encoder(wsi)

    def forward(self, wsi, mri):
        x_wsi = self.wsi_projector(self.encode_wsi(wsi))
        x_mri = self.mri_projector(self.mri_encoder(mri))

        wsi_mu, wsi_sigma, wsi_logvar = self.wsi_distribution(x_wsi)
        mri_mu, mri_sigma, mri_logvar = self.mri_distribution(x_mri)

        wsi_gated = self.wsi_gate(wsi_mu)
        mri_gated = self.mri_gate(mri_mu)
        fused = torch.cat([wsi_gated, mri_gated], dim=1)

        fused_logits = self.fusion_classifier(fused)
        weight = self.fusion_classifier.weight
        bias = self.fusion_classifier.bias
        half_bias = bias / 2.0 if bias is not None else None
        wsi_logits = F.linear(wsi_gated, weight[:, : wsi_gated.size(1)], half_bias)
        mri_logits = F.linear(mri_gated, weight[:, wsi_gated.size(1) :], half_bias)

        return {
            "logits": fused_logits,
            "wsi_logits": wsi_logits,
            "mri_logits": mri_logits,
            "wsi_mu": wsi_mu,
            "mri_mu": mri_mu,
            "wsi_sigma": wsi_sigma,
            "mri_sigma": mri_sigma,
            "wsi_logvar": wsi_logvar,
            "mri_logvar": mri_logvar,
            "wsi_gated": wsi_gated,
            "mri_gated": mri_gated,
        }

    def distribution_loss(self, outputs, lambda_reg=2e-2):
        align = 0.5 * (
            gaussian_kl(
                outputs["mri_mu"],
                outputs["mri_logvar"],
                outputs["wsi_mu"],
                outputs["wsi_logvar"],
            )
            + gaussian_kl(
                outputs["wsi_mu"],
                outputs["wsi_logvar"],
                outputs["mri_mu"],
                outputs["mri_logvar"],
            )
        )
        reg = 0.5 * (
            standard_normal_kl(outputs["wsi_mu"], outputs["wsi_logvar"])
            + standard_normal_kl(outputs["mri_mu"], outputs["mri_logvar"])
        )
        return align + lambda_reg * reg, align, reg

    def compute_loss(self, outputs, labels, criterion, alpha=0.2, lambda_reg=2e-2):
        cls_loss = criterion(outputs["logits"], labels)
        dist_loss, align_loss, reg_loss = self.distribution_loss(outputs, lambda_reg=lambda_reg)
        total = (1.0 - alpha) * cls_loss + alpha * dist_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "dist_loss": dist_loss.detach(),
            "align_loss": align_loss.detach(),
            "reg_loss": reg_loss.detach(),
        }
