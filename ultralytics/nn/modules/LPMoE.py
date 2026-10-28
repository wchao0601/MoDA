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

from .ms_deform_attn import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .adapter_modules_LPMoE import SpatialPriorModule, InteractionBlock, deform_inputs
from torchvision import transforms

# from aim.v2.utils import load_pretrained
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算到dinov3-main的路径
dinov3_path = os.path.join(current_dir, '..', '..', '..', 'dinov3-main')
dinov3_path = os.path.abspath(dinov3_path)

# 添加到Python路径
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

# 现在可以导入
from dinov3.models.vision_transformer import DinoVisionTransformer

dino_backbones = {
    "small": {"name": "dinov3_vits16", "embedding_size": 384, "patch_size": 16},
    "base": {"name": "dinov3_vitb16", "embedding_size": 768, "patch_size": 16},
    "large": {"name": "dinov3_vitl16", "embedding_size": 1024, "patch_size": 16},
}

intermediate_layers = {
    "small": [2, 5, 8, 11],
    "base": [2, 5, 8, 11],
    "large": [4, 11, 17, 23],
}

weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
}

class LPMoE(nn.Module):
    def __init__(self, in_chanels=3, embed_dim=768, size="base", conv_inplane=64, drop_rate=0., img_size=224, deform_num_heads=6, n_points=4, init_values=0., 
                with_cffn=True, drop_path_rate=0., cffn_ratio=0.25, deform_ratio=1.0, interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], 
                use_extra_extractor=True, add_vit_feature=True):

        super(LPMoE, self).__init__()
        self.size = size
        self.backbone = load("facebookresearch/dinov3", dino_backbones[self.size]["name"], pretrained=None)
        state_dict = torch.load(weight_path[self.size])
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.embed_dim = embed_dim
        self.interaction_indexes = interaction_indexes
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.n = intermediate_layers[self.size]
        self.add_vit_feature = add_vit_feature
        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.level_embed1 = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        self.spm1 = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=False)
            for i in range(len(interaction_indexes))
        ])
        self.interactions1 = nn.Sequential(*[
            InteractionBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=False)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)
        self.norm11 = nn.SyncBatchNorm(embed_dim)
        self.norm21 = nn.SyncBatchNorm(embed_dim)
        self.norm31 = nn.SyncBatchNorm(embed_dim)
        self.norm41 = nn.SyncBatchNorm(embed_dim)

        self.num_tokens = 1
        self.img_size = (img_size, img_size)
        num_patches = self.backbone.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.up1.apply(self._init_weights)
        self.spm1.apply(self._init_weights)
        self.interactions1.apply(self._init_weights)

        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _add_level_embed1(self, c2, c3, c4):
        c2 = c2 + self.level_embed1[0]
        c3 = c3 + self.level_embed1[1]
        c4 = c4 + self.level_embed1[2]
        return c2, c3, c4

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.img_size[0] // 16, self.img_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward(self, feature):
        # with torch.no_grad():
        x, y = feature
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / 16, x.shape[3] / 16)

        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x) #[1,384,64,64] [1,1024,384] [1,256,384] [1,64,384]
        c2, c3, c4 = self._add_level_embed(c2, c3, c4) # [1,1024,384] [1,256,384] [1,64,384]
        c = torch.cat([c2, c3, c4], dim=1) # [1,1344,384]
        bs, n, dim = c3.shape

        # Patch Embedding forward
        x = self.backbone.patch_embed(x)  # [1,16,16,384]
        if len(x.shape) == 4:
            H, W = x.shape[1:3]  # 直接获取 H 和 W
            H, W = int(H), int(W)
        else:
            L = x.shape[1]       # 获取序列长度 L
            H = W = int(math.sqrt(L))  # 计算平方根得到 H 和 W
        x = x.flatten(1, 2) # [1,256,384]
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W) # [1,256,384]
        x = self.pos_drop(x + pos_embed) # [1,256,384]


        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.backbone.blocks[indexes[0]:indexes[-1] + 1],
                        deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
        

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)


        # y
        deform_inputs11, deform_inputs21 = deform_inputs(y)

        # SPM forward
        c11, c21, c31, c41 = self.spm1(y) #[1,384,64,64] [1,1024,384] [1,256,384] [1,64,384]
        c21, c31, c41 = self._add_level_embed(c21, c31, c41) # [1,1024,384] [1,256,384] [1,64,384]
        cy = torch.cat([c21, c31, c41], dim=1) # [1,1344,384]
        bs1, n1, dim1 = c31.shape

        # Patch Embedding forward
        y = self.backbone.patch_embed(y)  # [1,16,16,384]
        if len(y.shape) == 4:
            H, W = y.shape[1:3]  # 直接获取 H 和 W
            H, W = int(H), int(W)
        else:
            L = x.shape[1]       # 获取序列长度 L
            H = W = int(math.sqrt(L))  # 计算平方根得到 H 和 W
        y = y.flatten(1, 2) # [1,256,384]
        bs, n, dim = y.shape
        pos_embed1 = self._get_pos_embed(self.pos_embed1[:, 1:], H, W) # [1,256,384]
        y = self.pos_drop(y + pos_embed1) # [1,256,384]


        # Interaction
        outs1 = list()
        for i, layer in enumerate(self.interactions1):
            indexes = self.interaction_indexes[i]
            y, cy = layer(y, cy, self.backbone.blocks[indexes[0]:indexes[-1] + 1],
                        deform_inputs11, deform_inputs21, H, W)
            outs1.append(y.transpose(1, 2).view(bs, dim, H, W).contiguous())
        

        # Split & Reshape
        c21 = cy[:, 0:c21.size(1), :]
        c31 = cy[:, c21.size(1):c21.size(1) + c31.size(1), :]
        c41 = cy[:, c21.size(1) + c31.size(1):, :]

        c21 = c21.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c31 = c31.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c41 = c41.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c11 = self.up1(c21) + c11

        if self.add_vit_feature:
            y1, y2, y3, y4 = outs1
            y1 = F.interpolate(y1, scale_factor=4, mode='bilinear', align_corners=False)
            y2 = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=False)
            y4 = F.interpolate(y4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c11, c21, c31, c41 = c11 + y1, c21 + y2, c31 + y3, c41 + y4

        # Final Norm
        f11 = self.norm11(c11)
        f21 = self.norm21(c21)
        f31 = self.norm31(c31)
        f41 = self.norm41(c41)

        return [f2, f3, f4, f21, f31, f41]


