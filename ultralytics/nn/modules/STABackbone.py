import torch
import sys
import os
import warnings
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.nn.parameter import Parameter
from torch.hub import load
from safetensors import torch as sf_torch # 导入 safetensors 的 torch 模块
from .conv import Conv
from .block import C3k2, SPPF, C2PSA
from copy import deepcopy
from torch import Tensor
import logging
from .ms_deform_attn import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .my_adapter_modules import SpatialPriorModule, InteractionBlock1, deform_inputs
from torchvision import transforms
from .sfs_conv import SFS_Conv
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
dinov3_path = os.path.join(current_dir, '..', '..', '..', 'dinov3-main')
dinov3_path = os.path.abspath(dinov3_path)
from timm.models.layers import DropPath

if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)
from dinov3.models.vision_transformer import DinoVisionTransformer

logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")


dino_backbones = {
    "small": {"name": "dinov3_vits16", "embedding_size": 384, "patch_size": 16},
    "base": {"name": "dinov3_vitb16", "embedding_size": 768, "patch_size": 16},
    "large": {"name": "dinov3_vitl16", "embedding_size": 1024, "patch_size": 16},
}

intermediate_layers = {
    "small": [5, 8, 11],
    "base": [5, 8, 11],
    "large": [11, 17, 23],
}

weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
}

images_norm = {
    "small": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "base": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "large": {"mean": (0.430, 0.411, 0.296), "std": (0.213, 0.156, 0.143)},
}

class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4

class STABackbone(nn.Module): 
    def __init__(
        self, in_chanels=3, embed_dim=384, size="base", conv_inplane=64
    ):
        super().__init__()
        self.emb_dim = embed_dim
        self.conv_inplane = conv_inplane
        self.size = size
        self.backbone = load("facebookresearch/dinov3", dino_backbones[self.size]["name"], pretrained=None)
        state_dict = torch.load(weight_path[self.size])
        self.intermediate_layers = intermediate_layers[self.size]
        self.backbone.load_state_dict(state_dict, strict=False)

        del self.backbone.head
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.mlcextractor = SpatialPriorModulev2(inplanes=self.conv_inplane)  
        self.mlcextractor1 = SpatialPriorModulev2(inplanes=self.conv_inplane)     

        # linear projection
        hidden_dim = self.emb_dim
        # hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(self.emb_dim + self.conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.emb_dim + self.conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.emb_dim + self.conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.emb_dim + self.conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.emb_dim + self.conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.emb_dim + self.conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])
        self.input_norm = v2.Normalize(mean=images_norm[self.size]["mean"], std=images_norm[self.size]["std"]) # sat493m
    

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x, y = feature
        x = self.input_norm(x)
        y = self.input_norm(y)
        
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // 16, x.shape[3] // 16
        bs, _, _, _ = x.shape

        all_layers = self.backbone.get_intermediate_layers(x, n=self.intermediate_layers, return_class_token=True)
        all_layers1 = self.backbone.get_intermediate_layers(y, n=self.intermediate_layers, return_class_token=True)

        sem_feats, sem_feats1 = [], []
        num_scales = len(all_layers) - 2
        for i, (sem_feat, sem_feat1) in enumerate(zip(all_layers, all_layers1)):
            feat, _ = sem_feat
            feat1, _ = sem_feat1
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            sem_feat1 = feat1.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feat1 = F.interpolate(sem_feat1, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)
            sem_feats1.append(sem_feat1)

        # fusion
        fused_feats, fused_feats1 = [], []
        detail_feats = self.mlcextractor(x)
        detail_feats1 = self.mlcextractor1(y)
        for sem_feat, detail_feat in zip(sem_feats, detail_feats):
            fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        for sem_feat, detail_feat in zip(sem_feats1, detail_feats1):
            fused_feats1.append(torch.cat([sem_feat, detail_feat], dim=1))

        x2 = self.norms[0](self.convs[0](fused_feats[0]))
        x3 = self.norms[1](self.convs[1](fused_feats[1]))
        x4 = self.norms[2](self.convs[2](fused_feats[2]))
        y2 = self.norms[3](self.convs[3](fused_feats1[0]))
        y3 = self.norms[4](self.convs[4](fused_feats1[1]))
        y4 = self.norms[5](self.convs[5](fused_feats1[2]))

        return [x2, x3, x4, y2, y3, y4]

