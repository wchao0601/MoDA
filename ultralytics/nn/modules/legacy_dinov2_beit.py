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
from .beit import BEiTBaseline
from torchvision import transforms
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
dinov2_path = os.path.join(current_dir, '..', '..', '..', 'dinov2')
dinov2_path = os.path.abspath(dinov2_path)
from timm.models.layers import DropPath

if dinov2_path not in sys.path:
    sys.path.insert(0, dinov2_path)
try:
    from dinov2.models.vision_transformer import DinoVisionTransformer, vit_small_p16, vit_base_p16, vit_large_p16
except ModuleNotFoundError:
    # Loading a serialized checkpoint does not invoke the constructor, so the
    # original DINOv2 package is only required when building a fresh model.
    DinoVisionTransformer = None
    vit_small_p16 = vit_base_p16 = vit_large_p16 = None

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




BEiT_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 16},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 16},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 16},
}

BEiT_weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/deit_small_patch16_224-cd65a155.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/deit_base_patch16_224-b5f2ef4d.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth',
}

BEiT_intermediate_layers = {
    "small": [5, 8, 11],
    "base": [5, 8, 11],
    "large": [11, 17, 23],
}

class BEiTBackbone(nn.Module):
    def __init__(self, in_chanels=3, embed_dim=768, size="base", conv_inplane=64):
        from torchvision import transforms

        super(BEiTBackbone, self).__init__()
        self.emb_dim = embed_dim
        self.size = size
        self.conv_inplane = conv_inplane
        if size == 'small':
            self.pretrain_size = (224,224)
            self.backbone = BEiTBaseline(patch_size=16,embed_dim=384,depth=12,num_heads=6,mlp_ratio=4,drop_path_rate=0.2) # load("facebookresearch/dinov2", dinov2_backbones[self.size]["name"], pretrained=None)
        elif size == 'base':
            self.pretrain_size = (224,224)
            self.backbone = BEiTBaseline(patch_size=16,embed_dim=768,depth=12,num_heads=12,mlp_ratio=4,drop_path_rate=0.3) 
        elif size == 'large':
            self.pretrain_size = (384,384)
            self.backbone = BEiTBaseline(img_size=384,patch_size=16,embed_dim=1024,depth=24,num_heads=16,mlp_ratio=4,drop_path_rate=0.4)

        state_dict = torch.load(BEiT_weight_path[self.size])
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.mlcextractor = MLCExtractor(inplanes=self.conv_inplane)  
        self.mlcextractor1 = MLCExtractor(inplanes=self.conv_inplane)  
        self.twoadapter = MoRAdapter(num_experts=4, hidden_size = self.emb_dim)
        for name, module in self.twoadapter.named_children():
            for param in module.parameters():
                    param.requires_grad = True

        self.inet_norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.n = BEiT_intermediate_layers[self.size]

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
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim)
        ])

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed[:,1:,:].reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward_features(self, x, y):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.backbone.patch_embed(x)
        y, _ = self.backbone.patch_embed(y)
        batch_size, seq_len, _ = x.size()
        
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        if self.backbone.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.backbone.pos_embed, Hp, Wp)
            x = x + pos_embed
            y = y + pos_embed

        x = torch.cat((cls_tokens, x), dim=1)
        y = torch.cat((cls_tokens, y), dim=1)
        x = self.backbone.pos_drop(x)
        y = self.backbone.pos_drop(y)
        
        rel_pos_bias = self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None
        features = []
        features1 = []
        for i, blk in enumerate(self.backbone.blocks):
            
            x = blk(x, rel_pos_bias)
            y = blk(y, rel_pos_bias)
            if i in self.n:
                x, y = self.twoadapter(x,y)
                features.append(x)
                features1.append(y)

        return tuple(features), tuple(features1)

    def forward(self, feature):
        x, y = feature
        bs, _, _, _ = x.shape

        x = self.inet_norm(x)
        y = self.inet_norm(y)

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        all_layers, all_layers1 = self.forward_features(x, y)

        fused_feats, fused_feats1 = [], []
        detail_feats = self.mlcextractor(x)
        detail_feats1 = self.mlcextractor1(y)

        sem_feats, sem_feats1 = [], []
        # sl_feats, sl_feats1 = [], []
        num_scales = len(all_layers) - 2
        # scales_ratio = [2.0, 1.0, 0.5]
        for i, (sem_feat, sem_feat1) in enumerate(zip(all_layers, all_layers1)):
            feat = sem_feat
            feat1 = sem_feat1
            sem_feat = feat[:, 1:, :].transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            sem_feat1 = feat1[:, 1:, :].transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feat1 = F.interpolate(sem_feat1, size=[resize_H, resize_W], mode="bilinear", align_corners=False)

            sem_feats.append(sem_feat)
            sem_feats1.append(sem_feat1)


        # fusion detail_feats[i].shape[2], detail_feats[i].shape[3] 
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


dinov2_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 16},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 16},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 16},
}

weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/dinov2_vits14_reg4_pretrain_14to16.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/dinov2_vitb14_reg4_pretrain_14to16.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/dinov2_vitl14_reg4_pretrain_14to16.pth',
}

dinov2_intermediate_layers = {
    "small": [5, 8, 11],
    "base": [5, 8, 11],
    "large": [11, 17, 23],
}

class DINOv2Backbone(nn.Module):
    def __init__(self, in_chanels=3, embed_dim=768, size="base", conv_inplane=64):
        from torchvision import transforms

        super(DINOv2Backbone, self).__init__()
        self.emb_dim = embed_dim
        self.size = size
        self.conv_inplane = conv_inplane
        if size == 'small':
            self.backbone = vit_small_p16() # load("facebookresearch/dinov2", dinov2_backbones[self.size]["name"], pretrained=None)
        elif size == 'base':
            self.backbone = vit_base_p16() 
        elif size == 'large':
            self.backbone = vit_large_p16()

        state_dict = torch.load(weight_path[self.size])
        self.backbone.load_state_dict(state_dict, strict=False)
        del self.backbone.head
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.mlcextractor = MLCExtractor(inplanes=self.conv_inplane)  
        self.mlcextractor1 = MLCExtractor(inplanes=self.conv_inplane)  
        self.twoadapter = MoRAdapter(num_experts=4, hidden_size = self.emb_dim)
        for name, module in self.twoadapter.named_children():
            for param in module.parameters():
                    param.requires_grad = True

        self.inet_norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.n = dinov2_intermediate_layers[self.size]

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
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim)
        ])

    def transform(self, x):
        # x should have shape (B, C, H, W)
        b, c, h, w = x.shape

        # Compute how many pixels to drop to make height/width multiples of 14
        h_new = h - (h % 14)
        w_new = w - (w % 14)

        dh = h - h_new  # total pixels to drop in height
        dw = w - w_new  # total pixels to drop in width

        # Decide how many pixels to drop from each side.
        # For simplicity, drop half from each edge if dh/dw are even.
        # If dh or dw are odd, we'll drop one extra pixel from the "bottom/right".
        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left

        # Crop the tensor from edges
        # Make sure slicing indices do not go negative
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]

        # normalize with imagenet mean and std
        x_cropped = self.inet_norm(x_cropped)

        return x_cropped

    def _get_intermediate_layers_not_chunked(self, x, y, n=1):
        x = self.backbone.prepare_tokens_with_masks(x)
        y = self.backbone.prepare_tokens_with_masks(y)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, output1, total_block_len = [], [], len(self.backbone.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            y = blk(y)
            if i in blocks_to_take:
                x, y = self.twoadapter(x,y)
                output.append(x)
                output1.append(y)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output, output1

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
       
        outputs, outputs1 = self._get_intermediate_layers_not_chunked(x, y, n)
        if norm:
            outputs = [self.backbone.norm(out) for out in outputs]
            outputs1 = [self.backbone.norm(out1) for out1 in outputs1]
        class_tokens = [out[:, 0] for out in outputs]
        class_tokens1 = [out1[:, 0] for out1 in outputs1]
        outputs = [out[:, 1 + self.backbone.num_register_tokens :] for out in outputs]
        outputs1 = [out1[:, 1 + self.backbone.num_register_tokens :] for out1 in outputs1]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.backbone.patch_size, h // self.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
            outputs1 = [
                out1.reshape(B, w // self.backbone.patch_size, h // self.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out1 in outputs1
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens)), tuple(zip(outputs1, class_tokens1))
        return tuple(outputs), tuple(outputs1)

    def forward(self, feature):
        x, y = feature
        bs, _, _, _ = x.shape

        x = self.inet_norm(x)
        y = self.inet_norm(y)

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        all_layers, all_layers1 = self.get_intermediate_layers(x, y, n=self.n, return_class_token=True)

        fused_feats, fused_feats1 = [], []
        detail_feats = self.mlcextractor(x)
        detail_feats1 = self.mlcextractor1(y)

        sem_feats, sem_feats1 = [], []
        # sl_feats, sl_feats1 = [], []
        num_scales = len(all_layers) - 2
        # scales_ratio = [2.0, 1.0, 0.5]
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


        # fusion detail_feats[i].shape[2], detail_feats[i].shape[3] 
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



class MLCExtractor(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()
        # 1/4
        self.stem = nn.Sequential(
            *[
                MLConv(3, inplanes, k=3, s=2),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.mlconv2 = nn.Sequential(
            *[
                MLConv(inplanes, 2 * inplanes, k=3, s=2),
                nn.BatchNorm2d(2 * inplanes),
            ]
        )
        # 1/16
        self.mlconv3 = nn.Sequential(
            *[
                nn.ReLU(),
                MLConv(2 * inplanes, 4 * inplanes, k=3, s=2),
                nn.BatchNorm2d(4 * inplanes),
            ]
        )
        # 1/32
        self.mlconv4 = nn.Sequential(
            *[
                nn.ReLU(),
                MLConv(4 * inplanes, 4 * inplanes, k=3, s=2),
                nn.BatchNorm2d(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.mlconv2(c1)     # 1/8
        c3 = self.mlconv3(c2)     # 1/16
        c4 = self.mlconv4(c3)     # 1/32

        return c2, c3, c4


class LoRAdapter(nn.Module):
    def __init__(self, dim=512, rank=8):
        super(LoRAdapter, self).__init__()
        self.adapter_down = nn.Linear(dim, rank)
        self.adapter_mid = nn.Linear(rank, rank)
        self.drop = nn.Dropout(0.1)
        self.adapter_up = nn.Linear(rank, dim)

    def forward(self, x):
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.drop(x_down)
        x_up = self.adapter_up(x_down)
        return x_up


class MBIA(nn.Module):
    # MoE-based Bidirectional Interactive Adapter
    def __init__(self, num_experts=8, dim=512):
        super(MBIA, self).__init__()
        self.num_experts = num_experts
        self.drop = nn.Dropout(0.1)
        self.experts = nn.ModuleList([LoRAdapter(dim, rank=8) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gating_scores = self.gate(x)
        gating_weights = F.softmax(gating_scores, dim=-1)   # [b, n, num]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [b, n, d, num]
        output = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=-1)
        return output


class MoRAdapter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(MoRAdapter, self).__init__()
        self.norm_text = nn.LayerNorm(hidden_size)
        self.norm_image = nn.LayerNorm(hidden_size)

        # cross-modality adapter
        self.adapter_text = MBIA(num_experts, hidden_size)
        self.adapter_image = MBIA(num_experts, hidden_size)

        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, text, image):
        # --- text branch ---
        text_out = text + self.adapter_text(image)          # 跨模态注入
        text_norm = self.norm_text(text)
        text_embeds = text_out + self.ffn(text_norm)        # FFN增强

        # --- image branch ---
        image_norm = self.norm_image(image)
        image_out = image + self.adapter_image(text)        # 跨模态注入
        image_embeds = image_out + self.ffn(image_norm)     # FFN增强

        return text_embeds, image_embeds


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        """Apply convolution to input tensor."""
        return self.conv(x) # self.act(self.bn(self.conv(x)))


class MLConv(nn.Module):  
    ''' MoE-based Lightning Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        p = [(k + 2, 0, 1, 0), (0, k + 2, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k + 2), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

        # 初始化门控网络
        self.gating_network = ExpertSelector(in_features=c1, experts=4)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # 全局池化后将输入降维用于门控网络
        x_flattened = x.view(batch_size, channels, -1)
        gate_weights = self.gating_network(x_flattened.mean(dim=-1))  # 输出 (batch_size, 4) 的权重

        yw0 = gate_weights[:, 0].view(-1, 1, 1, 1) * self.cw(self.pad[0](x))
        yw1 = gate_weights[:, 1].view(-1, 1, 1, 1) * self.cw(self.pad[1](x))
        yh0 = gate_weights[:, 2].view(-1, 1, 1, 1) * self.ch(self.pad[2](x))
        yh1 = gate_weights[:, 3].view(-1, 1, 1, 1) * self.ch(self.pad[3](x))
        
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class ExpertSelector(nn.Module):
    def __init__(self, in_features,experts):
        super(ExpertSelector, self).__init__()
        self.fc = nn.Linear(in_features, experts)  # 输出3个专家的权重
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)  # 归一化为概率分布


class GetDinoLayer(nn.Module):
    def __init__(self, in_chanels=384, out_channels=384, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return x[0], x[3]
        elif self.index == 1:
            return x[1], x[4]
        elif self.index == 2:
            return x[2], x[5]


class BAConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class A_Attn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = BAConv(dim, all_head_dim * 3, 1, act=False)
        self.qkv1 = BAConv(dim, all_head_dim * 3, 1, act=False)
        self.proj = BAConv(all_head_dim, dim, 1, act=False)
        self.pe = BAConv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)


    def forward(self, x, x1):
        """Processes the input tensor 'x' through the area-attention"""
        B1, C1, H1, W1 = x.shape
        if H1 % self.area != 0:
            pad1 = self.area - (H1 % self.area)
            x = F.pad(x, pad=(0, pad1, 0, pad1))
            x1 = F.pad(x1, pad=(0, pad1, 0, pad1))
            B, C, H, W = x.shape
            N = H * W
        else:
            B, C, H, W = x.shape
            N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        qkv1 = self.qkv1(x1).flatten(2).transpose(1, 2)

        qkv = qkv + qkv1

        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )

        if x.is_cuda and USE_FLASH_ATTN:
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        elif x.is_cuda and not USE_FLASH_ATTN:
            x = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=None, dropout_p=0.0, is_causal=False)
            x = x.permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 3, 1)
            k = k.permute(0, 2, 3, 1)
            v = v.permute(0, 2, 3, 1)
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values 
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))
            x = x.permute(0, 3, 1, 2)
            v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = x + self.pe(v)
        x = self.proj(x)

        x = x[:, :, :H1, :W1]
        return x



class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, mlp_ratio=2.0, qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        num_heads= int(dim / 32)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.cpe1_1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.norm1_1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.in_proj_1 = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.act_proj_1 = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwc_1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.act_1 = nn.SiLU()
        self.attn = A_Attn(dim, num_heads=num_heads, area=2) # Area Attention
        # self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)

        # 可学习模态权重
        self.gate_rgb = nn.Parameter(torch.tensor(0.5))
        self.gate_sar = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb, sar):
        x = rgb
        x1 = sar
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x1 = x1.flatten(2).permute(0, 2, 1)
        B, L, C = x.shape

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x1 = x1 + self.cpe1_1(x1.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        shortcut1 = x1
        x = self.norm1(x)
        x1 = self.norm1_1(x1)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x1 = self.in_proj_1(x1).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2)))
        x1 = self.act_1(self.dwc_1(x1.permute(0, 3, 1, 2)))

        # Area Attention
        x = self.attn(x, x1).permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = self.out_proj(x * act_res)
        x = self.gate_rgb*shortcut + self.gate_sar*shortcut1 + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FFM(nn.Module):  # Feature Fusion Module (AFM)
    """Concatenate a list of tensors along dimension."""
    
    def __init__(self, c1):
        super(FFM, self).__init__()
        self.mlla = MLLABlock(dim=c1)
    def forward(self,x):
        fuse = self.mlla(x[0], x[1])
        return fuse
