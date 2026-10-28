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

class LoRABackbone(nn.Module): 
    def __init__(
        self, in_chanels=3, embed_dim=384, size="base", conv_inplane=64, r=8
    ):
        super().__init__()
        self.emb_dim = embed_dim
        self.conv_inplane = conv_inplane
        self.size = size
        self.use_lora = True
        self.backbone = load("facebookresearch/dinov3", dino_backbones[self.size]["name"], pretrained=None)
        state_dict = torch.load(weight_path[self.size])
        self.intermediate_layers = intermediate_layers[self.size]
        self.backbone.load_state_dict(state_dict, strict=False)

        del self.backbone.head
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.backbone1 = deepcopy(self.backbone)
        self.input_norm = v2.Normalize(mean=images_norm[self.size]["mean"], std=images_norm[self.size]["std"]) # sat493m
    
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []
            self.w_a1 = []
            self.w_b1 = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
            )

            for i, block in enumerate(self.backbone1.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a1.extend([w_a_linear_q, w_a_linear_v])
                self.w_b1.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
            )

            self._reset_lora_parameters()


    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)
        for w_a in self.w_a1:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b1:
            nn.init.zeros_(w_b.weight)


    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x, y = feature
        x = self.input_norm(x)
        y = self.input_norm(y)
        
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // 16, x.shape[3] // 16
        bs, _, _, _ = x.shape

        all_layers = self.backbone.get_intermediate_layers(x, n=self.intermediate_layers, return_class_token=True)
        all_layers1 = self.backbone1.get_intermediate_layers(y, n=self.intermediate_layers, return_class_token=True)

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

        x2, x3, x4 = sem_feats
        y2, y3, y4 = sem_feats1

        return [x2, x3, x4, y2, y3, y4]

    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        w_a1, w_b1 = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}
            w_a1 = {f"w_a1_{i:03d}": self.w_a1[i].weight for i in range(len(self.w_a1))}
            w_b1 = {f"w_b1_{i:03d}": self.w_b1[i].weight for i in range(len(self.w_a1))}

        torch.save({**w_a, **w_b, **w_a1, **w_b1}, filename)

    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        # Load the LoRA parameters
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)

            for i, w_A_linear in enumerate(self.w_a1):
                saved_key = f"w_a1_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b1):
                saved_key = f"w_b1_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)


class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv
