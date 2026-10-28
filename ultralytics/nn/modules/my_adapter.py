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
from .conv import Conv
from .block import C3k2, SPPF, C2PSA
from copy import deepcopy
from torch import Tensor

from .ms_deform_attn import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .my_adapter_modules import SpatialPriorModule, InteractionBlock1, deform_inputs
from torchvision import transforms
from .sfs_conv import SFS_Conv
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

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
    "giant": {"name": "dinov3_vitg16", "embedding_size": 1536, "patch_size": 16},
}

intermediate_layers = {
    "small": [2, 5, 8, 11],
    "base": [2, 5, 8, 11],
    "large": [4, 11, 17, 23],
    "giant": [9, 19, 29, 39],
}

weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
}


class SARPriorModule1(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.conv1 = Conv(c1=3, c2=64, k=3, s=2)
        self.conv2 = Conv(c1=64, c2=128, k=3, s=2)
        self.c3k2_1 = SFS_Conv(c1=128, c2=384)
        self.conv3 = Conv(c1=384, c2=128, k=3, s=2)
        self.c3k2_2 = SFS_Conv(c1=128, c2=384)
        self.conv4 = Conv(c1=384, c2=128, k=3, s=2)
        self.c3k2_3 = SFS_Conv(c1=128, c2=384)
        self.conv5 = Conv(c1=384, c2=128, k=3, s=2)
        self.c3k2_4 = SFS_Conv(c1=128, c2=384)
        # self.sppf = SPPF(c1=384, c2=384)
        # self.c2psa = C2PSA(c1=384, c2=384)

        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x): # [1,3,256,256]

        c1 = self.conv1(x)   # [1,128,128,128]
        c1 = self.conv2(c1)  # [1,384,64,64]
        c1 = self.c3k2_1(c1) # [1,384,64,64]   
        c2 = self.conv3(c1)  # [1,384,32,32] 
        c2 = self.c3k2_2(c2) # [1,384,32,32]    
        c3 = self.conv4(c2)  # [1,384,16,16]
        c3 = self.c3k2_3(c3) # [1,384,16,16]
        c4 = self.conv5(c3)  # [1,384,8,8]
        c4 = self.c3k2_4(c4)
        # c4 = self.sppf(c4) 
        # c4 = self.c2psa(c4)

        bs, dim, _, _ = c1.shape
        # # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


class SARPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.conv1 = Conv(c1=3, c2=64, k=3, s=2)
        self.conv2 = Conv(c1=64, c2=128, k=3, s=2)
        self.c3k2_1 = C3k2(c1=128, c2=384, c3k=False, e=0.25)
        self.conv3 = Conv(c1=384, c2=128, k=3, s=2)
        self.c3k2_2 = C3k2(c1=128, c2=384, c3k=False, e=0.25)
        self.conv4 = Conv(c1=384, c2=128, k=3, s=2)
        self.c3k2_3 = C3k2(c1=128, c2=384, c3k=True)
        self.conv5 = Conv(c1=384, c2=128, k=3, s=2)
        self.c3k2_4 = C3k2(c1=128, c2=384, c3k=True)
        # self.sppf = SPPF(c1=384, c2=384)
        # self.c2psa = C2PSA(c1=384, c2=384)

        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x): # [1,3,256,256]

        c1 = self.conv1(x)   # [1,128,128,128]
        c1 = self.conv2(c1)  # [1,384,64,64]
        c1 = self.c3k2_1(c1) # [1,384,64,64]   
        c2 = self.conv3(c1)  # [1,384,32,32] 
        c2 = self.c3k2_2(c2) # [1,384,32,32]    
        c3 = self.conv4(c2)  # [1,384,16,16]
        c3 = self.c3k2_3(c3) # [1,384,16,16]
        c4 = self.conv5(c3)  # [1,384,8,8]
        c4 = self.c3k2_4(c4)
        # c4 = self.sppf(c4) 
        # c4 = self.c2psa(c4)

        bs, dim, _, _ = c1.shape
        # # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4

class SpatialPriorModule1(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4

class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                MoEPConv(3, inplanes, k=3, s=2),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                MoEPConv(inplanes, 2 * inplanes, k=3, s=2),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.ReLU(),
                MoEPConv(2 * inplanes, 4 * inplanes, k=3, s=2),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.ReLU(),
                MoEPConv(4 * inplanes, 4 * inplanes, k=3, s=2),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4

class MyAdapter(nn.Module):
    def __init__(
        self, in_chanels=3, embed_dim=384, size="base",
        r: int = 8,
        use_lora: bool = True,
        img_dim: tuple[int, int] = (256, 256),
    ):
        """The DINOv2 encoder-decoder model for finetuning to downstream tasks.

        Args:
            encoder (nn.Module): The ViT encoder model loaded with the DINOv2 model weights.
            r (int, optional): The rank parameter of the LoRA weights. Defaults to 3.
            emb_dim (int, optional): The embedding dimension of the encoder. Defaults to 1024.
            use_lora (bool, optional): Determines whether to use LoRA. Defaults to False.
            img_dim (tuple[int, int], optional): The input image dimension. Defaults to
                (520, 520).
        """
        super().__init__()
        self.emb_dim = embed_dim
        self.img_dim = img_dim
        self.size = size
        self.use_lora = use_lora
        self.backbone = load("facebookresearch/dinov3", dino_backbones[self.size]["name"], pretrained=None)
        state_dict = torch.load(weight_path[self.size])
        self.intermediate_layers = intermediate_layers[self.size]
        self.backbone.load_state_dict(state_dict, strict=False)
        assert img_dim[0] % self.backbone.patch_size == 0, "Wrong input shape for patches"
        assert r > 0

        del self.backbone.head
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.backbone1 = deepcopy(self.backbone)  
        conv_inplane = 64
        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)  
        self.sta1 = SpatialPriorModulev2(inplanes=conv_inplane)     
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.twoadapter = MoRAdapter(num_experts=4, hidden_size = embed_dim)

        for name, module in self.twoadapter.named_children():
            for param in module.parameters():
                    param.requires_grad = True

        # linear projection
        hidden_dim = embed_dim
        # hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

        self.convs1 = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms1 = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

            
    def transform(self, x):
        # x should have shape (B, C, H, W)
        b, c, h, w = x.shape
        h_new = h - (h % 16)
        w_new = w - (w % 16)
        dh = h - h_new  # total pixels to drop in height
        dw = w - w_new  # total pixels to drop in widthght".
        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left
        # Make sure slicing indices do not go negative
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]
        # normalize with imagenet mean and std
        x_cropped = self.inet_norm(x_cropped)
        return x_cropped


    def _get_intermediate_layers_not_chunked(self, x: Tensor, y: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)
        y, (H, W) = self.backbone1.prepare_tokens_with_masks(y)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, output1, total_block_len = [], [], len(self.backbone.blocks)
        # output.append(x)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, (blk, blk1) in enumerate(zip(self.backbone.blocks,self.backbone1.blocks)):
            if self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
                rope_sincos1 = self.backbone1.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
                rope_sincos1 = None
            x = blk(x, rope_sincos)
            y = blk1(y, rope_sincos1)
            if i in blocks_to_take:
                x,y = self.twoadapter(x,y)
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
                    x_norm_cls_reg1 = self.backbone1.cls_norm(out1[:, : self.backbone1.n_storage_tokens + 1])
                    x_norm_patch = self.backbone.norm(out[:, self.backbone.n_storage_tokens + 1 :])
                    x_norm_patch1 = self.backbone1.norm(out1[:, self.backbone1.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                    outputs_normed1.append(torch.cat((x_norm_cls_reg1, x_norm_patch1), dim=1))
                else:
                    outputs_normed.append(self.backbone.norm(out))
                    outputs_normed1.append(self.backbone1.norm(out1))
            outputs = outputs_normed
            outputs1 = outputs_normed1
        class_tokens = [out[:, 0] for out in outputs]
        class_tokens1 = [out1[:, 0] for out1 in outputs1]
        extra_tokens = [out[:, 1 : self.backbone.n_storage_tokens + 1] for out in outputs]
        extra_tokens1 = [out1[:, 1 : self.backbone1.n_storage_tokens + 1] for out1 in outputs1]
        outputs = [out[:, self.backbone.n_storage_tokens + 1 :] for out in outputs]
        outputs1 = [out1[:, self.backbone1.n_storage_tokens + 1 :] for out1 in outputs1]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.backbone.patch_size, w // self.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
            outputs1 = [
                out1.reshape(B, h // self.backbone1.patch_size, w // self.backbone1.patch_size, -1).permute(0, 3, 1, 2).contiguous()
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
        # x = self.transform(x) # [1,3,256,256]
        # y = self.transform(y) # [1,3,256,256]
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // 16, x.shape[3] // 16
        bs, C, h, w = x.shape

        all_layers, all_layers1 = self.get_intermediate_layers(x, y, n=[5,8,11], return_class_token=True)
        # all_layers1 = self.get_intermediate_layers1(y, n=[5,8,11], return_class_token=True)


        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)
        # fusion
        fused_feats = []
        detail_feats = self.sta(x)
        for sem_feat, detail_feat in zip(sem_feats, detail_feats):
            fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        sem_feats1 = []
        num_scales1 = len(all_layers1) - 2
        for i, sem_feat in enumerate(all_layers1):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales1-i)), int(W_c * 2**(num_scales1-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats1.append(sem_feat)
        # fusion
        fused_feats1 = []
        detail_feats1 = self.sta1(y)
        for sem_feat, detail_feat in zip(sem_feats1, detail_feats1):
            fused_feats1.append(torch.cat([sem_feat, detail_feat], dim=1))
        y2 = self.norms1[0](self.convs1[0](fused_feats1[0]))
        y3 = self.norms1[1](self.convs1[1](fused_feats1[1]))
        y4 = self.norms1[2](self.convs1[2](fused_feats1[2]))

        return [c2, c2, c3, c4, y2, y2, y3, y4]


class MoRAdapter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(MoRAdapter, self).__init__()
        self.norm_text = nn.LayerNorm(hidden_size)
        self.norm_image = nn.LayerNorm(hidden_size)

        # cross-modality adapter
        self.adapter_text = MoBA(num_experts, hidden_size)
        self.adapter_image = MoBA(num_experts, hidden_size)

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
        text_norm = self.norm_text(text)
        text_out = text + self.adapter_text(image)          # 跨模态注入
        text_embeds = text_out + self.ffn(text_norm)        # FFN增强

        # --- image branch ---
        image_norm = self.norm_image(image)
        image_out = image + self.adapter_image(text)        # 跨模态注入
        image_embeds = image_out + self.ffn(image_norm)     # FFN增强

        return text_embeds, image_embeds

class SimplifiedTwoAdapter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(SimplifiedTwoAdapter, self).__init__()
        self.hidden_size = hidden_size
        
        # 共享的层归一化和dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # 共享的注意力机制（可以处理两种模态）
        self.cross_attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        
        # 共享的FFN层
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # 简化的适配器 - 每个模态只需要一个交叉适配器
        self.cross_adapter_text = MoBA(num_experts, hidden_size)
        self.cross_adapter_image = MoBA(num_experts, hidden_size)

    def forward(self, text, image):
        # 第一步：自注意力 + 交叉适配器
        text_norm = self.norm(text)
        text_attn, _ = self.cross_attention(text_norm, text_norm, text_norm)
        text_attn = self.dropout(text_attn)
        
        image_norm = self.norm(image)
        image_attn, _ = self.cross_attention(image_norm, image_norm, image_norm)
        image_attn = self.dropout(image_attn)
        
        # 残差连接 + 交叉适配器注入
        text_out = text + text_attn + self.cross_adapter_text(image)
        image_out = image + image_attn + self.cross_adapter_image(text)
        
        # 第二步：FFN处理
        text_ffn = self.ffn(self.norm(text_out))
        image_ffn = self.ffn(self.norm(image_out))
        
        text_final = text_out + self.dropout(text_ffn)
        image_final = image_out + self.dropout(image_ffn)
        
        return text_final, image_final


class TwoAdapter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(TwoAdapter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.attn_text = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.attn_image = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.activation = nn.ReLU()
        self.linear_text1 = nn.Linear(hidden_size, hidden_size)
        self.linear_text2 = nn.Linear(hidden_size, hidden_size)
        self.linear_image1 = nn.Linear(hidden_size, hidden_size)
        self.linear_image2 = nn.Linear(hidden_size, hidden_size)

        self.adapter_text1 = MoBA(num_experts, hidden_size)
        self.adapter_text2 = MoBA(num_experts, hidden_size)
        self.adapter_image1 = MoBA(num_experts, hidden_size)
        self.adapter_image2 = MoBA(num_experts, hidden_size)

    def forward(self, text, image):
        text_norm = self.norm(text)
        t_att, _ = self.attn_text(text_norm, text_norm, text_norm)
        t_att = self.dropout(t_att)

        image_norm = self.norm(image)
        v_att, _ = self.attn_image(image_norm, image_norm, image_norm)
        v_att = self.dropout(v_att)

        text_out = text + t_att + self.adapter_text1(image)
        image_out = image + v_att + self.adapter_image1(text)

        text_norm2 = self.norm(text_out)
        t_tmp = self.linear_text1(self.dropout(self.activation(self.linear_text2(self.norm(text_norm2)))))
        text_embeds = text_out + self.dropout(t_tmp)

        image_norm2 = self.norm(image_out)
        i_tmp = self.linear_image1(self.dropout(self.activation(self.linear_image2(self.norm(image_norm2)))))
        image_embeds = image_out + self.dropout(i_tmp)

        text_embeds = self.adapter_text2(image_out) + text_embeds
        image_embeds = self.adapter_image2(text_out) + image_embeds
        return text_embeds, image_embeds


class Adapter(nn.Module):
    def __init__(self, dim=512, rank=8):
        super(Adapter, self).__init__()
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

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.conv(x) # self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class MoEPConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k+2, 0, 1, 0), (0, k+2, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k+2), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        # 初始化门控网络
        self.gating_network = GatingNetwork(in_features=c1, experts=4)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # 全局池化后将输入降维用于门控网络
        x_flattened = x.view(batch_size, channels, -1)
        gate_weights = self.gating_network(x_flattened.mean(dim=-1))  # 输出 (batch_size, 3) 的权重

        yw0 = gate_weights[:, 0].view(-1, 1, 1, 1) * self.cw(self.pad[0](x))
        yw1 = gate_weights[:, 1].view(-1, 1, 1, 1) * self.cw(self.pad[1](x))
        yh0 = gate_weights[:, 2].view(-1, 1, 1, 1) * self.ch(self.pad[2](x))
        yh1 = gate_weights[:, 3].view(-1, 1, 1, 1) * self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

class GatingNetwork(nn.Module):
    def __init__(self, in_features,experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(in_features, experts)  # 输出3个专家的权重
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)  # 归一化为概率分布


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

class GetDinoLayer(nn.Module):
    def __init__(self, in_chanels=384, out_channels=384, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return x[0], x[4]
        elif self.index == 1:
            return x[1], x[5]
        elif self.index == 2:
            return x[2], x[6]
        elif self.index == 3:
            return x[3], x[7]

