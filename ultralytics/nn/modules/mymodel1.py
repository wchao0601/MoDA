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
from .block import C3k2, SPPF, C2PSA, AFM
from .fusion import CMIM # M4-SAR
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


from timm.models.layers import DropPath

current_dir = os.path.dirname(os.path.abspath(__file__))
dinov2_path = os.path.join(current_dir, '..', '..', '..', 'dinov2')
dinov2_path = os.path.abspath(dinov2_path)
from timm.models.layers import DropPath

if dinov2_path not in sys.path:
    sys.path.insert(0, dinov2_path)
from dinov2.models.vision_transformer import DinoVisionTransformer, vit_small_p16, vit_base_p16, vit_large_p16

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
    "giant": {"name": "dinov3_vitg16", "embedding_size": 1536, "patch_size": 16},
}

intermediate_layers = {
    "small": [5, 8, 11],
    "base": [5, 8, 11],
    "large": [11, 17, 23],
    "giant": [19, 29, 39],
}

weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
}


class E2E_OSDet_Backbone(nn.Module):
    def __init__(self, dim = 32):
        super().__init__()

        self.conv_rgb_1 = Conv(3, 32, 3, 2)
        self.conv_rgb_2 = Conv(32, 64, 3, 2)
        self.c3k2_rgb_1 = C3k2(64, 128, False, 0.25)
        self.conv_rgb_3 = Conv(128, 128, 3, 2)
        self.c3k2_rgb_2 = C3k2(128, 256, False, 0.25)

        self.conv_rgb_4 = Conv(256, 256, 3, 2)
        self.c3k2_rgb_3 = C3k2(256, 256, False, 0.25)
        self.conv_rgb_5 = Conv(256, 512, 3, 2)

        self.c3k2_rgb_4 = C3k2(512, 512, True)
        self.sppf_rgb_1 = SPPF(512, 512, 5)

        self.conv_sar_1 = Conv(3, 32, 3, 2)
        self.conv_sar_2 = Conv(32, 64, 3, 2)
        self.c3k2_sar_1 = C3k2(64, 128, False, 0.25)
        self.conv_sar_3 = Conv(128, 128, 3, 2)
        self.c3k2_sar_2 = C3k2(128, 256, False, 0.25)

        self.conv_sar_4 = Conv(256, 256, 3, 2)
        self.c3k2_sar_3 = C3k2(256, 256, False, 0.25)

        self.conv_sar_5 = Conv(256, 512, 3, 2)
        self.c3k2_sar_4 = C3k2(512, 512, True)
        self.sppf_sar_1 = SPPF(512, 512, 5)

        self.cmim_1 = CMIM(256, 16)
        self.cmim_2 = CMIM(256, 16)
        self.cmim_3 = CMIM(512, 32)

        self.afm_1 = AFM(256)
        self.afm_2 = AFM(256)
        self.afm_3 = AFM(512)

        self.s2psa = C2PSA(512,512)

    def forward(self, x, y):
        x1 = self.conv_rgb_1(x)
        x1 = self.conv_rgb_2(x1)
        x1 = self.c3k2_rgb_1(x1)
        x1 = self.conv_rgb_3(x1)
        x2 = self.c3k2_rgb_2(x1)

        y1 = self.conv_sar_1(y)
        y1 = self.conv_sar_2(y1)
        y1 = self.c3k2_sar_1(y1)
        y1 = self.conv_sar_3(y1)
        y2 = self.c3k2_sar_2(y1)

        f1, f2 = self.cmim_1([x2, y2])
        x2, y2 = x2 + f1, y2 + f2

        x3 = self.conv_rgb_4(x2)
        x3 = self.c3k2_rgb_3(x3)

        y3 = self.conv_sar_4(y2)
        y3 = self.c3k2_sar_3(y3)

        f3, f4 = self.cmim_2([x3, y3])
        x3, y3 = x3 + f3, y3 + f4

        x4 = self.conv_rgb_5(x3)
        x4 = self.c3k2_rgb_4(x4)
        x4 = self.sppf_rgb_1(x4)

        y4 = self.conv_sar_5(y3)
        y4 = self.c3k2_sar_4(y4)
        y4 = self.sppf_sar_1(y4)

        f5, f6 = self.cmim_3([x4, y4])
        x4, y4 = x4 + f5, y4 + f6

        c2 = self.afm_1([x2, y2])
        c3 = self.afm_2([x3, y3])
        c4 = self.afm_3([x4, y4])
        c4 = self.s2psa(c4)

        return c2, c3, c4

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

def load_pretrained_backbone(model_path='/opt/data/private/CODE/work4/MSRODet1/runs/ogsod-2.0/v11s-E2E-OSDet-300e/weights/best.pt', n=36):
    """在函数中加载，避免循环导入"""
    from ultralytics import YOLO
    model = YOLO(model_path).model.model[:n]
    # 提取backbone（前6-10层，根据具体版本调整）
    # backbone_layers = []
    # for i, (name, module) in enumerate(model.named_children()):
    #     if i < 8:  # 调整这个数字
    #         backbone_layers.append(module)
    return model # nn.Sequential(*backbone_layers)

def load_pretrained_backbone1(model_path='/opt/data/private/CODE/work4/MSRODet1/runs/ogsod-2.0/v11s-E2E-OSDet-300e/weights/best.pt', n=36):
    from ultralytics import YOLO
    model = YOLO(model_path).model  # 获取 PyTorch 模型部分
    model.train()  # 设置为训练模式
    # 存储中间特征的列表
    features = []
    # 定义钩子函数
    def hook_fn(module, input, output):
        features.append(output)

    # 注册钩子（需要先找到第25,26,27层）
    # YOLOv8 模型结构比较复杂，需要先探索层索引
    print("Model structure:")
    for i, (name, module) in enumerate(model.named_children()):
        print(f"{i}: {name} - {type(module)}")

    # 根据实际结构注册钩子（这里假设conv层）
    # 可能需要根据实际模型结构调整索引
    hook_handles = []
    hook_layers = [25, 26, 27]  # 要提取的层索引

    for i, (name, module) in enumerate(model.named_modules()):
        if i in hook_layers:
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
            print(f"Registered hook at layer {i} ({name})")
    
    return model



class OSDetExtractor(nn.Module):
    def __init__(self, backbone, dim = 32):
        super().__init__()
        self.Silence = backbone[0] 
        self.FAM_OPT = backbone[1] 
        self.FAM_SAR = backbone[2] 

        self.conv_rgb_1 = backbone[3] 
        self.conv_rgb_2 = backbone[4] 
        self.c3k2_rgb_1 = backbone[5] 
        self.conv_rgb_3 = backbone[6] 
        self.c3k2_rgb_2 = backbone[7] 

        self.conv_sar_1 = backbone[8] 
        self.conv_sar_2 = backbone[9] 
        self.c3k2_sar_1 = backbone[10] 
        self.conv_sar_3 = backbone[11] 
        self.c3k2_sar_2 = backbone[12] 

        self.cmim_1 = backbone[13] 

        self.conv_rgb_4 = backbone[16] 
        self.c3k2_rgb_3 = backbone[17] 
        self.conv_sar_4 = backbone[18] 
        self.c3k2_sar_3 = backbone[19] 

        self.cmim_2 = backbone[20]

        self.conv_rgb_5 = backbone[23] 
        self.c3k2_rgb_4 = backbone[24] 
        self.sppf_rgb_1 = backbone[25] 
        self.conv_sar_5 = backbone[26] 
        self.c3k2_sar_4 = backbone[27] 
        self.sppf_sar_1 = backbone[28] 

        self.cmim_3 = backbone[29] 

        self.afm_1 = backbone[32] 
        self.afm_2 = backbone[33] 
        self.afm_3 = backbone[34] 

        self.s2psa = backbone[35] 

    def forward(self, feature):
        feature = self.Silence(feature)
        x = self.FAM_OPT(feature)
        x1 = self.conv_rgb_1(x)
        x1 = self.conv_rgb_2(x1)
        x1 = self.c3k2_rgb_1(x1)
        x1 = self.conv_rgb_3(x1)
        x2 = self.c3k2_rgb_2(x1)

        y = self.FAM_SAR(feature)
        y1 = self.conv_sar_1(y)
        y1 = self.conv_sar_2(y1)
        y1 = self.c3k2_sar_1(y1)
        y1 = self.conv_sar_3(y1)
        y2 = self.c3k2_sar_2(y1)

        f1, f2 = self.cmim_1([x2, y2])
        x2, y2 = x2 + f1, y2 + f2

        x3 = self.conv_rgb_4(x2)
        x3 = self.c3k2_rgb_3(x3)

        y3 = self.conv_sar_4(y2)
        y3 = self.c3k2_sar_3(y3)

        f3, f4 = self.cmim_2([x3, y3])
        x3, y3 = x3 + f3, y3 + f4

        x4 = self.conv_rgb_5(x3)
        x4 = self.c3k2_rgb_4(x4)
        x4 = self.sppf_rgb_1(x4)

        y4 = self.conv_sar_5(y3)
        y4 = self.c3k2_sar_4(y4)
        y4 = self.sppf_sar_1(y4)

        f5, f6 = self.cmim_3([x4, y4])
        x4, y4 = x4 + f5, y4 + f6

        c2 = self.afm_1([x2, y2])
        c3 = self.afm_2([x3, y3])
        c4 = self.afm_3([x4, y4])
        c4 = self.s2psa(c4)

        return c2, c3, c4

class CSSAExtractor(nn.Module):
    def __init__(self, backbone, dim = 32):
        super().__init__()
        self.Silence = backbone[0] 
        self.FAM_OPT = backbone[1] 
        self.FAM_SAR = backbone[2] 

        self.conv_rgb_1 = backbone[3] 
        self.conv_rgb_2 = backbone[4] 
        self.c3k2_rgb_1 = backbone[5] 
        self.conv_rgb_3 = backbone[6] 
        self.c3k2_rgb_2 = backbone[7] 
        self.conv_rgb_4 = backbone[8] 
        self.c3k2_rgb_3 = backbone[9] 
        self.conv_rgb_5 = backbone[10] 
        self.c3k2_rgb_4 = backbone[11] 


        self.conv_sar_1 = backbone[12] 
        self.conv_sar_2 = backbone[13] 
        self.c3k2_sar_1 = backbone[14] 
        self.conv_sar_3 = backbone[15] 
        self.c3k2_sar_2 = backbone[16] 
        self.conv_sar_4 = backbone[17] 
        self.c3k2_sar_3 = backbone[18] 
        self.conv_sar_5 = backbone[19] 
        self.c3k2_sar_4 = backbone[20] 


        self.icafusion_1 = backbone[21] 
        self.icafusion_2 = backbone[22] 
        self.icafusion_3 = backbone[23] 

        self.sppf = backbone[24]
        self.c2psa = backbone[25]

    def forward(self, feature):
        feature = self.Silence(feature)
        x = self.FAM_OPT(feature)
        y = self.FAM_SAR(feature)

        x1 = self.conv_rgb_1(x) 
        x1 = self.conv_rgb_2(x1)
        x1 = self.c3k2_rgb_1(x1)
        x2 = self.conv_rgb_3(x1)
        x2 = self.c3k2_rgb_2(x2)
        x3 = self.conv_rgb_4(x2)
        x3 = self.c3k2_rgb_3(x3)
        x4 = self.conv_rgb_5(x3)
        x4 = self.c3k2_rgb_4(x4)


        y1 = self.conv_sar_1(y) 
        y1 = self.conv_sar_2(y1)
        y1 = self.c3k2_sar_1(y1)
        y2 = self.conv_sar_3(y1)
        y2 = self.c3k2_sar_2(y2)
        y3 = self.conv_sar_4(y2)
        y3 = self.c3k2_sar_3(y3)
        y4 = self.conv_sar_5(y3)
        y4 = self.c3k2_sar_4(y4)


        c2 = self.icafusion_1([x2, y2])
        c3 = self.icafusion_2([x3, y3])
        c4 = self.icafusion_3([x4, y4])
        c4 = self.sppf(c4)
        c4 = self.c2psa(c4)

        return c2, c3, c4

class CMADetExtractor(nn.Module):
    def __init__(self, backbone, dim = 32):
        super().__init__()
        self.Silence = backbone[0] 
        self.FAM_OPT = backbone[1] 
        self.FAM_SAR = backbone[2] 

        self.conv_rgb_1 = backbone[3] 
        self.conv_rgb_2 = backbone[4] 
        self.c3k2_rgb_1 = backbone[5] 
        self.conv_sar_1 = backbone[6] 
        self.conv_sar_2 = backbone[7] 
        self.c3k2_sar_1 = backbone[8] 
        self.osr_0 = backbone[9] 

        self.conv_rgb_3 = backbone[12] 
        self.c3k2_rgb_2 = backbone[13] 
        self.conv_sar_3 = backbone[14] 
        self.c3k2_sar_2 = backbone[15] 
        self.osr_1 = backbone[16] 


        self.conv_rgb_4 = backbone[19] 
        self.c3k2_rgb_3 = backbone[20] 
        self.conv_sar_4 = backbone[21] 
        self.c3k2_sar_3 = backbone[22] 
        self.osr_2 = backbone[23]

        self.conv_rgb_5 = backbone[26] 
        self.c3k2_rgb_4 = backbone[27] 
        self.sppf_rgb_1 = backbone[28] 
        self.conv_sar_5 = backbone[29] 
        self.c3k2_sar_4 = backbone[30] 
        self.sppf_sar_1 = backbone[31] 
        self.osr_3 = backbone[32] 

        self.afm_1 = backbone[35] 
        self.afm_2 = backbone[36] 
        self.afm_3 = backbone[37] 

        self.s2psa = backbone[38] 

    def forward(self, feature):
        feature = self.Silence(feature)
        x = self.FAM_OPT(feature)
        y = self.FAM_SAR(feature)

        x = self.conv_rgb_1(x)
        x = self.conv_rgb_2(x)
        x = self.c3k2_rgb_1(x)
        y = self.conv_sar_1(y)
        y = self.conv_sar_2(y)
        y = self.c3k2_sar_1(y)
        f0, f7 = self.osr_0([x, y])
        x, y = x + f0, y + f7

        x1 = self.conv_rgb_3(x)
        x2 = self.c3k2_rgb_2(x1)
        y1 = self.conv_sar_3(y)
        y2 = self.c3k2_sar_2(y1)

        f1, f2 = self.osr_1([x2, y2])
        x2, y2 = x2 + f1, y2 + f2

        x3 = self.conv_rgb_4(x2)
        x3 = self.c3k2_rgb_3(x3)

        y3 = self.conv_sar_4(y2)
        y3 = self.c3k2_sar_3(y3)

        f3, f4 = self.osr_2([x3, y3])
        x3, y3 = x3 + f3, y3 + f4

        x4 = self.conv_rgb_5(x3)
        x4 = self.c3k2_rgb_4(x4)
        x4 = self.sppf_rgb_1(x4)

        y4 = self.conv_sar_5(y3)
        y4 = self.c3k2_sar_4(y4)
        y4 = self.sppf_sar_1(y4)

        f5, f6 = self.osr_3([x4, y4])
        x4, y4 = x4 + f5, y4 + f6

        c2 = self.afm_1([x2, y2])
        c3 = self.afm_2([x3, y3])
        c4 = self.afm_3([x4, y4])
        c4 = self.s2psa(c4)

        return c2, c3, c4

dinov2_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 16},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 16},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 16},
}

dinov2_weight_path = {
    "small": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/dinov2_vits14_reg4_pretrain_14to16.pth',
    "base": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/dinov2_vitb14_reg4_pretrain_14to16.pth',
    "large": '/opt/data/private/CODE/work6/yolo-dinov2-1/weights/dinov2_vitl14_reg4_pretrain_14to16.pth',
}

dinov2_intermediate_layers = {
    "small": [5, 8, 11],
    "base": [5, 8, 11],
    "large": [11, 17, 23],
}


class mybackbone(nn.Module): 
    def __init__(
        self, in_chanels=3, embed_dim=384, size="base", conv_inplane=64
    ):
        super().__init__()
        self.emb_dim = embed_dim
        self.conv_inplane = conv_inplane
        self.size = size
        self.backbone = load("facebookresearch/dinov3", dino_backbones[self.size]["name"], pretrained=None)
        # if size == 'small':
        #     self.backbone = vit_small_p16() # load("facebookresearch/dinov2", dinov2_backbones[self.size]["name"], pretrained=None)
        # elif size == 'base':
        #     self.backbone = vit_base_p16() 
        # elif size == 'large':
        #     self.backbone = vit_large_p16()
        state_dict = torch.load(weight_path[self.size])
        self.intermediate_layers = intermediate_layers[self.size]
        self.backbone.load_state_dict(state_dict, strict=False)

        del self.backbone.head
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.multibackbone = load_pretrained_backbone(model_path="/opt/data/private/CODE/work4/multi-ultralytics/runs/msod/v11s-CSSA-300e/weights/best.pt", n=39) #(model_path="/opt/data/private/CODE/work4/MSRODet1/runs/M4-SAR/yolo11-obb-E2E-OSDet-area4/weights/best.pt") # E2E_OSDet_Backbone(dim = self.conv_inplane)
        self.twomlcextractor = CSSAExtractor(backbone = self.multibackbone) # ICAFusionExtractor(backbone = multibackbone) # OSDetExtractor(backbone = multibackbone)
        for name, module in self.twomlcextractor.named_children():
            for param in module.parameters():
                    param.requires_grad = True

        # linear projection
        hidden_dim = self.emb_dim
        # hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.fusion = GatedAttentionFusion(self.emb_dim + 256, hidden_dim)
        self.fusion1 = GatedAttentionFusion(self.emb_dim + 256, hidden_dim)
        self.fusion2 = GatedAttentionFusion(self.emb_dim + 512, hidden_dim)
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(self.emb_dim + 256, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Conv2d(self.emb_dim + 256, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Conv2d(self.emb_dim + 512, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # ])
        # # norm
        # self.norms = nn.ModuleList([
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.BatchNorm2d(hidden_dim)
        # ])

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x, y = feature
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // 16, x.shape[3] // 16
        bs, _, _, _ = x.shape
        all_layers = self.backbone.get_intermediate_layers(x, n=[5,8,11], return_class_token=True)
        all_layers1 = self.backbone.get_intermediate_layers(y, n=[5,8,11], return_class_token=True)

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
        fused_feats = []
        fused_feats1 = []
        detail_feats = self.twomlcextractor(torch.cat([x, y],dim=1))

        for sem_feat, detail_feat in zip(sem_feats, detail_feats):
            fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        
        # for sem_feat1, detail_feat in zip(sem_feats1, detail_feats):
        #     fused_feats1.append(torch.cat([sem_feat1, detail_feat], dim=1))

        x2 = self.fusion(fused_feats[0])
        x3 = self.fusion1(fused_feats[1])
        x4 = self.fusion2(fused_feats[2])
        # x2 = self.norms[0](self.convs[0](fused_feats[0]))
        # x3 = self.norms[1](self.convs[1](fused_feats[1]))
        # x4 = self.norms[2](self.convs[2](fused_feats[2]))
        y2, y3, y4 = sem_feats1

        return [x2, x3, x4, y2, y3, y4]

class GatedAttentionFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.qkv = nn.Linear(c1, c1)  
        # Gating mechanism
        self.gate = nn.Linear(c1, c1)
        self.out_proj = nn.Linear(c1, c2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1) 
        # 3. Compute gate scores
        gate_scores = torch.sigmoid(self.gate(x))
        
        # 4. Apply gating
        gated_out = x * gate_scores
        
        # 5. Concatenate heads and project
        output = self.out_proj(gated_out).permute(0, 3, 1, 2)
        
        return output


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
        output = torch.sum(gating_weights.unsqueeze(3) * expert_outputs, dim=-1)
        return output

class MoRAdapter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(MoRAdapter, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # cross-modality adapter
        self.adapter1 = MBIA(num_experts, hidden_size)
        self.adapter2 = MBIA(num_experts, hidden_size)

        # feed-forward network
        # self.ffn = SwiGLUFFN(hidden_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, x, x1):
        x = x.permute(0, 2, 3, 1)
        x1 = x1.permute(0, 2, 3, 1)
        # --- x branch ---
        x_norm = self.norm1(x)
        x_out = x + self.adapter1(x1)          # 跨模态注入
        x_embeds = x_out + self.ffn(x_norm)        # FFN增强

        # --- x1 branch ---
        x1_norm = self.norm2(x1)
        x1_out = x1 + self.adapter2(x)        # 跨模态注入
        x1_embeds = x1_out + self.ffn(x1_norm)     # FFN增强

        return x_embeds.permute(0, 3, 1, 2), x1_embeds.permute(0, 3, 1, 2)


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


class GetLayer(nn.Module):
    def __init__(self, in_chanels=384, out_channels=384, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return x[0]
        elif self.index == 1:
            return x[1]
        elif self.index == 2:
            return x[2]


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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # 可学习模态权重
        self.gate_rgb = nn.Parameter(torch.tensor(0.5))
        self.gate_sar = nn.Parameter(torch.tensor(0.5))

        self.convs = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, bias=False)
        # norm
        self.norms = nn.BatchNorm2d(dim)
    

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
        # act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x1 = self.in_proj_1(x1).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2)))
        x1 = self.act_1(self.dwc_1(x1.permute(0, 3, 1, 2)))

        # Area Attention
        # x = self.attn(x, x1).permute(0, 2, 3, 1).contiguous().view(B, L, C)
        x = self.norms(self.convs(torch.cat([x,x1],dim=1))).permute(0, 2, 3, 1).contiguous().view(B, L, C)
        act_res = self.act(self.act_proj(x))

        x = self.out_proj(x * act_res)
        x = self.gate_rgb*shortcut + self.gate_sar*shortcut1  + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FFM1(nn.Module):  # Feature Fusion Module (AFM)
    """Concatenate a list of tensors along dimension."""
    
    def __init__(self, c1):
        super(FFM1, self).__init__()
        self.mlla = MLLABlock(dim=c1)
    def forward(self,x):
        fuse = self.mlla(x[0], x[1])
        return fuse


