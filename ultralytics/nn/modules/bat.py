import torch
import sys
import os
import warnings
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.hub import load
from safetensors import torch as sf_torch # 导入 safetensors 的 torch 模块
from copy import deepcopy
from torch import Tensor
import logging
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .my_adapter_modules import SpatialPriorModule, InteractionBlock1, deform_inputs
from torchvision import transforms
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from timm.models.layers import DropPath



dino_backbones = {
    "small": {"name": "dinov3_vits16", "embedding_size": 384, "patch_size": 16},
    "base": {"name": "dinov3_vitb16", "embedding_size": 768, "patch_size": 16},
}

intermediate_layers = {
    "small": [5, 8, 11],
    "base": [5, 8, 11],
}

weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
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

class BATBackbone(nn.Module): 
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

        self.twoadapter = MoRAdapter(num_experts=8, hidden_size = self.emb_dim)

        for name, module in self.twoadapter.named_children():
            for param in module.parameters():
                    param.requires_grad = True

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

    def _get_intermediate_layers_not_chunked(self, x: Tensor, y: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)
        y, (H, W) = self.backbone.prepare_tokens_with_masks(y)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, output1, total_block_len = [], [], len(self.backbone.blocks)
        # output.append(x)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.backbone.blocks):
            if self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            y = blk(y, rope_sincos)
            if i in blocks_to_take:
                x1,x2 = torch.split(x,[5,x.shape[1]-5],dim=1)
                y1,y2 = torch.split(y,[5,x.shape[1]-5],dim=1)
                x2,y2 = self.twoadapter(x2,y2)
                x = torch.cat([x1,x2],dim=1)
                y = torch.cat([y1,y2],dim=1)
                output.append(x)
                output1.append(y)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output, output1

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs, outputs1 = self._get_intermediate_layers_not_chunked(x, y, n)
        if norm:
            outputs_normed, outputs_normed1 = [], []
            for (out, out1) in zip(outputs, outputs1):
                if self.backbone.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.backbone.cls_norm(out[:, : self.backbone.n_storage_tokens + 1])
                    x_norm_cls_reg1 = self.backbone.cls_norm(out1[:, : self.backbone.n_storage_tokens + 1])
                    x_norm_patch = self.backbone.norm(out[:, self.backbone.n_storage_tokens + 1 :])
                    x_norm_patch1 = self.backbone.norm(out1[:, self.backbone.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                    outputs_normed1.append(torch.cat((x_norm_cls_reg1, x_norm_patch1), dim=1))
                else:
                    outputs_normed.append(self.backbone.norm(out))
                    outputs_normed1.append(self.backbone.norm(out1))
            outputs = outputs_normed
            outputs1 = outputs_normed1
        class_tokens = [out[:, 0] for out in outputs]
        class_tokens1 = [out1[:, 0] for out1 in outputs1]
        extra_tokens = [out[:, 1 : self.backbone.n_storage_tokens + 1] for out in outputs]
        extra_tokens1 = [out1[:, 1 : self.backbone.n_storage_tokens + 1] for out1 in outputs1]
        outputs = [out[:, self.backbone.n_storage_tokens + 1 :] for out in outputs]
        outputs1 = [out1[:, self.backbone.n_storage_tokens + 1 :] for out1 in outputs1]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.backbone.patch_size, w // self.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
            outputs1 = [
                out1.reshape(B, h // self.backbone.patch_size, w // self.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out1 in outputs1
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs), tuple(outputs1)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens)), tuple(zip(outputs1, class_tokens1))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens)), tuple(zip(outputs1, extra_tokens1))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens)), tuple(zip(outputs1, class_tokens1, extra_tokens1))


    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x, y = feature
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // 16, x.shape[3] // 16
        bs, _, _, _ = x.shape
        all_layers, all_layers1 = self.get_intermediate_layers(x, y, n=[5,8,11], return_class_token=True)

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

        # x2, x3, x4 = sem_feats
        # y2, y3, y4 = sem_feats1
        return [x2, x3, x4, y2, y3, y4]




class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, r=8):
        super().__init__()

        self.adapter_down = nn.Linear(dim, r)  
        self.adapter_up = nn.Linear(r, dim)  
        self.adapter_mid = nn.Linear(r, r)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up

class MoRAdapter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(MoRAdapter, self).__init__()
        self.norm_text = nn.LayerNorm(hidden_size)
        self.norm_image = nn.LayerNorm(hidden_size)

        # cross-modality adapter
        self.adapter_text = Bi_direct_adapter(r=8, dim=hidden_size)# MoBA(num_experts, hidden_size)
        self.adapter_image = Bi_direct_adapter(r=8, dim=hidden_size)# MoBA(num_experts, hidden_size)
        
        # feed-forward network
        # self.ffn = SwiGLUFFN(hidden_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, text, image):
        # --- text branch ---
        text_norm = self.norm_text(text)
        text_out = text + self.adapter_text(image)          # 跨模态注入
        text_embeds = text_out + self.ffn(text_norm)        # FFN增强

        # --- image branch ---
        image_norm = self.norm_image(image)
        image_out = image + self.adapter_image(text)        # 跨模态注入
        image_embeds = image_out + self.ffn(image_norm)     # FFN增强

        return text_embeds, image_embeds


class Adapter(nn.Module):
    def __init__(self, dim=512, rank=8):
        super(Adapter, self).__init__()
        self.adapter_down = nn.Linear(dim, rank)
        self.adapter_up = nn.Linear(rank, dim)
        self.adapter_mid = nn.Linear(rank, rank)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.drop(x_down)
        x_up = self.adapter_up(x_down)
        return x_up


class MoBA(nn.Module):
    def __init__(self, num_experts=8, dim=512):
        super(MoBA, self).__init__()
        self.num_experts = num_experts
        self.drop = nn.Dropout(0.1)
        self.experts = nn.ModuleList([Adapter(dim, rank=8) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gating_scores = self.gate(x)
        gating_weights = F.softmax(gating_scores, dim=-1)   # [b, n, num]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [b, n, d, num]
        output = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=-1)
        return output
