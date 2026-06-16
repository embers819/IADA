import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def initialize_weights(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def region_partition(x, region_size):
    bsz, height, width, channels = x.shape
    x = x.view(
        bsz,
        height // region_size,
        region_size,
        width // region_size,
        region_size,
        channels,
    )
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return regions.view(-1, region_size, region_size, channels)


def region_reverse(regions, region_size, height, width):
    bsz = int(regions.shape[0] / (height * width / region_size / region_size))
    x = regions.view(
        bsz,
        height // region_size,
        width // region_size,
        region_size,
        region_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(bsz, height, width, -1)


class InnerAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        epeg=True,
        epeg_k=15,
        epeg_bias=True,
        epeg_type="attn",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.epeg_type = epeg_type

        if epeg:
            padding = epeg_k // 2
            if epeg_type == "attn":
                self.pe = nn.Conv2d(
                    num_heads,
                    num_heads,
                    (epeg_k, 1),
                    padding=(padding, 0),
                    groups=num_heads,
                    bias=epeg_bias,
                )
            else:
                self.pe = nn.Conv2d(
                    head_dim * num_heads,
                    head_dim * num_heads,
                    (epeg_k, 1),
                    padding=(padding, 0),
                    groups=head_dim * num_heads,
                    bias=epeg_bias,
                )
        else:
            self.pe = None

    def forward(self, x):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            bsz, num_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.pe is not None and self.epeg_type == "attn":
            attn = attn + self.pe(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == "value_bf":
            side = int(np.ceil(np.sqrt(num_tokens)))
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(bsz, channels, side, side))
            v = v + pe.reshape(bsz, self.num_heads, self.head_dim, num_tokens).permute(0, 1, 3, 2)

        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, -1)

        if self.pe is not None and self.epeg_type == "value_af":
            side = int(np.ceil(np.sqrt(num_tokens)))
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(bsz, channels, side, side))
            out = out + pe.reshape(bsz, -1, num_tokens).transpose(-1, -2)

        return self.proj_drop(self.proj(out))


class RegionAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        region_size=0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        region_num=8,
        epeg=False,
        min_region_num=0,
        min_region_ratio=0.0,
        **kwargs,
    ):
        super().__init__()
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        self.attn = InnerAttention(
            dim=dim,
            head_dim=head_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            epeg=epeg,
            **kwargs,
        )

    def _pad(self, x):
        bsz, length, channels = x.shape
        height = width = int(np.ceil(np.sqrt(length)))
        if self.region_size is not None:
            pad_side = -height % self.region_size
            height = width = height + pad_side
            region_size = self.region_size
            region_num = int(height // region_size)
        else:
            pad_side = -height % self.region_num
            height = width = height + pad_side
            region_num = self.region_num
            region_size = max(1, int(height // region_num))

        add_length = height * width - length
        if add_length > length / (self.min_region_ratio + 1e-8) or length < self.min_region_num:
            height = width = int(np.ceil(np.sqrt(length)))
            height = width = height + (-height % 2)
            add_length = height * width - length
            region_size = height
            region_num = 1

        if add_length > 0:
            x = torch.cat([x, x.new_zeros((bsz, add_length, channels))], dim=1)
        return x, height, width, add_length, region_num, region_size

    def forward(self, x):
        bsz, _, channels = x.shape
        x, height, width, add_length, _, region_size = self._pad(x)
        x = x.view(bsz, height, width, channels)
        x_regions = region_partition(x, region_size).view(-1, region_size * region_size, channels)
        x_regions = self.attn(x_regions)
        x_regions = x_regions.view(-1, region_size, region_size, channels)
        x = region_reverse(x_regions, region_size, height, width).view(bsz, height * width, channels)
        if add_length > 0:
            x = x[:, :-add_length]
        return x


class CrossRegionAttention(RegionAttention):
    def __init__(self, *args, crmsa_k=3, crmsa_mlp=False, **kwargs):
        super().__init__(*args, **kwargs)
        dim = kwargs.get("dim")
        if dim is None and len(args) > 0:
            dim = args[0]
        self.crmsa_mlp = crmsa_mlp
        self.crmsa_k = crmsa_k
        if crmsa_mlp:
            self.phi = nn.Sequential(
                nn.Linear(dim, dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(dim // 4, crmsa_k, bias=False),
            )
        else:
            self.phi = nn.Parameter(torch.empty(dim, crmsa_k))
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x):
        bsz, _, channels = x.shape
        x, height, width, add_length, _, region_size = self._pad(x)
        x = x.view(bsz, height, width, channels)
        x_regions = region_partition(x, region_size).view(-1, region_size * region_size, channels)

        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)
        else:
            logits = torch.einsum("wpc,cn->wpn", x_regions, self.phi).transpose(1, 2)

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)
        logits_min = logits.min(dim=-1).values
        logits_max = logits.max(dim=-1).values
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (
            logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8
        )

        pooled = torch.einsum("wpc,wnp->wnpc", x_regions, combine_weights).sum(dim=-2)
        pooled = self.attn(pooled.transpose(0, 1)).transpose(0, 1)
        x_regions = torch.einsum("wnc,wnp->wnpc", pooled, dispatch_weights_mm)
        x_regions = torch.einsum("wnpc,wnp->wpc", x_regions, dispatch_weights)

        x_regions = x_regions.view(-1, region_size, region_size, channels)
        x = region_reverse(x_regions, region_size, height, width).view(bsz, height * width, channels)
        if add_length > 0:
            x = x[:, :-add_length]
        return x


class TransLayer(nn.Module):
    def __init__(
        self,
        dim=512,
        head=8,
        drop_out=0.1,
        drop_path=0.0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        attn="rmsa",
        n_region=8,
        epeg=False,
        region_size=0,
        min_region_num=0,
        min_region_ratio=0.0,
        qkv_bias=True,
        crmsa_k=3,
        **kwargs,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if ffn else nn.Identity()
        if attn == "rmsa":
            self.attn = RegionAttention(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                **kwargs,
            )
        elif attn == "crmsa":
            self.attn = CrossRegionAttention(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported RRT attention: {attn}")

        act_layer = nn.GELU if ffn_act == "gelu" else nn.ReLU
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn = ffn
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), act_layer, drop_out) if ffn else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RRTEncoder(nn.Module):
    def __init__(
        self,
        mlp_dim=512,
        pos="none",
        attn="rmsa",
        region_num=8,
        drop_out=0.1,
        n_layers=2,
        n_heads=8,
        drop_path=0.0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        epeg=True,
        min_region_num=0,
        min_region_ratio=0.0,
        qkv_bias=True,
        cr_msa=True,
        crmsa_k=3,
        all_shortcut=False,
        crmsa_mlp=False,
        crmsa_heads=8,
        activation_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        if pos not in ("none", None):
            raise ValueError("The minimal IADA release keeps RRT-MIL with pos='none'.")
        self.final_dim = mlp_dim
        self.activation_checkpoint = activation_checkpoint
        self.all_shortcut = all_shortcut
        self.layers = nn.Sequential(
            *[
                TransLayer(
                    dim=mlp_dim,
                    head=n_heads,
                    drop_out=drop_out,
                    drop_path=drop_path,
                    ffn=ffn,
                    ffn_act=ffn_act,
                    mlp_ratio=mlp_ratio,
                    trans_dim=trans_dim,
                    attn=attn,
                    n_region=region_num,
                    epeg=epeg,
                    min_region_num=min_region_num,
                    min_region_ratio=min_region_ratio,
                    qkv_bias=qkv_bias,
                    **kwargs,
                )
                for _ in range(max(0, n_layers - 1))
            ]
        )
        self.cr_msa = (
            TransLayer(
                dim=mlp_dim,
                head=crmsa_heads,
                drop_out=drop_out,
                drop_path=drop_path,
                ffn=ffn,
                ffn_act=ffn_act,
                mlp_ratio=mlp_ratio,
                trans_dim=trans_dim,
                attn="crmsa",
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                crmsa_mlp=crmsa_mlp,
                **kwargs,
            )
            if cr_msa
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(mlp_dim)

    def _forward_layer(self, layer, x):
        if self.training and self.activation_checkpoint and x.requires_grad:
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x):
        original_ndim = x.ndim
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"RRTEncoder expects [N, C] or [B, N, C], got {tuple(x.shape)}")

        shortcut = x
        for layer in self.layers:
            x = self._forward_layer(layer, x)
        x = self._forward_layer(self.cr_msa, x)
        if self.all_shortcut:
            x = x + shortcut
        x = self.norm(x)
        return x.squeeze(0) if original_ndim == 2 else x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim=512, act="relu", gated=False, bias=False, dropout=False):
        super().__init__()
        hidden_dim = 128
        act_layer = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(act, nn.ReLU)
        if gated:
            self.attn_a = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), act_layer())
            self.attn_b = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.Sigmoid())
            self.attn_c = nn.Linear(hidden_dim, 1, bias=bias)
            self.gated = True
        else:
            layers = [nn.Linear(input_dim, hidden_dim, bias=bias), act_layer()]
            if dropout:
                layers.append(nn.Dropout(0.25))
            layers.append(nn.Linear(hidden_dim, 1, bias=bias))
            self.attn = nn.Sequential(*layers)
            self.gated = False

    def forward(self, x, return_attn=False):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if self.gated:
            logits = self.attn_c(self.attn_a(x) * self.attn_b(x))
        else:
            logits = self.attn(x)
        attn = F.softmax(logits.transpose(-1, -2), dim=-1)
        pooled = torch.matmul(attn, x).squeeze(1)
        if return_attn:
            return pooled, attn.squeeze(1)
        return pooled


class RRTMIL(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        mlp_dim=512,
        act="relu",
        n_classes=2,
        dropout=0.25,
        pos="none",
        attn="rmsa",
        pool="attn",
        region_num=8,
        n_layers=2,
        n_heads=8,
        drop_path=0.0,
        da_act="relu",
        trans_dropout=0.1,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        da_gated=False,
        da_bias=False,
        da_dropout=False,
        trans_dim=64,
        epeg=True,
        min_region_num=0,
        qkv_bias=True,
        fc=False,
        **kwargs,
    ):
        super().__init__()
        act_layer = nn.ReLU if act == "relu" else nn.GELU
        self.patch_to_emb = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim,
            pos=pos,
            attn=attn,
            region_num=region_num,
            n_layers=n_layers,
            n_heads=n_heads,
            drop_path=drop_path,
            drop_out=trans_dropout,
            ffn=ffn,
            ffn_act=ffn_act,
            mlp_ratio=mlp_ratio,
            trans_dim=trans_dim,
            epeg=epeg,
            min_region_num=min_region_num,
            qkv_bias=qkv_bias,
            **kwargs,
        )
        if pool == "attn":
            self.pool_fn = AttentionPooling(
                self.online_encoder.final_dim,
                da_act,
                gated=da_gated in (True, "gate"),
                bias=da_bias,
                dropout=da_dropout,
            )
        elif pool == "mean":
            self.pool_fn = None
        else:
            raise ValueError(f"Unsupported RRTMIL pool: {pool}")
        self.predictor = nn.Linear(mlp_dim, n_classes) if fc else nn.Identity()
        self.apply(initialize_weights)

    def forward(self, x, return_attn=False):
        x = self.patch_to_emb(x.float())
        x = self.online_encoder(x)
        if self.pool_fn is None:
            pooled = x.mean(dim=1 if x.ndim == 3 else 0, keepdim=False)
            attn = None
        else:
            if return_attn:
                pooled, attn = self.pool_fn(x, return_attn=True)
            else:
                pooled = self.pool_fn(x)
                attn = None
        logits_or_features = self.predictor(pooled)
        if return_attn:
            return logits_or_features, attn
        return logits_or_features
