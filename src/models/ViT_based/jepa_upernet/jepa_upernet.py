import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 假设 vision_transformer.py 就在同级目录下
from .vision_transformer import vit_small, vit_base, vit_large, vit_huge, VIT_EMBED_DIMS


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PPM(nn.Module):
    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        self.features = nn.ModuleList()
        for scale in pool_scales:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels, channels, 1)
            ))

    def forward(self, x):
        out = [x]
        for f in self.features:
            pool_out = f(x)
            pool_out = F.interpolate(pool_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            out.append(pool_out)
        return out


class SimpleUPerHead(nn.Module):
    def __init__(self, in_channels_list, channels, pool_scales=(1, 2, 3, 6), num_classes=1):
        super().__init__()
        self.psp_modules = PPM(pool_scales, in_channels_list[-1], channels)
        self.bottleneck = ConvModule(in_channels_list[-1] + len(pool_scales) * channels, channels, 3, padding=1)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:
            self.lateral_convs.append(ConvModule(in_channels, channels, 1))
            self.fpn_convs.append(ConvModule(channels, channels, 3, padding=1))
        self.fpn_bottleneck = ConvModule(len(in_channels_list) * channels, channels, 3, padding=1)
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def psp_forward(self, x):
        psp_outs = self.psp_modules(x)
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs[-1]))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear',
                                                              align_corners=False)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return self.cls_seg(feats)


class JEPA_UPerNet(nn.Module):
    def __init__(self, config, num_classes=1):
        super().__init__()
        self.config = config

        # 从 YACS 配置中读取参数
        self.model_name = config.MODEL.JEPA.MODEL_NAME
        self.patch_size = config.MODEL.JEPA.PATCH_SIZE
        self.embed_dim = config.MODEL.JEPA.EMBED_DIM
        self.out_indices = config.MODEL.JEPA.OUT_INDICES
        self.img_size = config.DATA.IMG_SIZE
        self.pool_scales = config.MODEL.UPERNET.POOL_SCALES

        # 1. 动态加载 Backbone
        if self.model_name == 'vit_small':
            self.backbone = vit_small(patch_size=self.patch_size, in_chans=3)
        elif self.model_name == 'vit_base':
            self.backbone = vit_base(patch_size=self.patch_size, in_chans=3)
        elif self.model_name == 'vit_large':
            self.backbone = vit_large(patch_size=self.patch_size, in_chans=3)
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported yet.")

        # 2. FPN 颈部
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3. UPerHead
        self.decode_head = SimpleUPerHead(
            in_channels_list=[self.embed_dim] * 4,
            channels=self.embed_dim,
            pool_scales=self.pool_scales,
            num_classes=num_classes
        )

        # 4. 执行规范化初始化
        self._init_weights()

    def _init_weights(self):
        """
        对模型所有层进行符合学术规范的初始化：
        - 卷积层：Kaiming Normal (He Init)，适用于 ReLU 系列激活函数
        - 归一化层：Weight=1, Bias=0
        - 全连接层：Truncated Normal
        """
        print("Initializing model weights with standard protocols...")
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_from(self):
        """
        权重加载逻辑：
        - 如果配置中指定了路径，则加载预训练权重覆盖初始化值。
        - 如果路径为 None 或 'none'，则跳过，实现从零开始的全量微调。
        """
        pretrained_path = self.config.MODEL.PRETRAIN_CKPT
        if not pretrained_path or str(pretrained_path).lower() == 'none':
            print("\n" + "!"*50)
            print("WARNING: No pretrain checkpoint specified.")
            print("Encoder will be trained FROM SCRATCH using random initialization.")
            print("!"*50 + "\n")
            return

        if not os.path.exists(pretrained_path):
            print(f"Error: Pretrained path {pretrained_path} does not exist. Skipping load.")
            return

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # 1. 提取核心字典
        if 'target_encoder' in checkpoint:
            raw_state_dict = checkpoint['target_encoder']
        elif 'model' in checkpoint:
            raw_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            raw_state_dict = checkpoint['state_dict']
        else:
            raw_state_dict = checkpoint

        # 2. 清洗前缀，对齐键名
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('predictor'):
                continue
            # 去除 DDP 或常见封装前缀
            new_k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '')
            state_dict[new_k] = v

        # 3. 加载权重
        msg = self.backbone.load_state_dict(state_dict, strict=False)

        # 找到 jepa_upernet.py 里面的这部分代码，修改打印逻辑：
        print(f"\n==================================================")
        print(f"Loaded JEPA weights successfully from {pretrained_path}.")

        # 1. 打印 Missing keys (你当前模型有，但预训练权重里没有的)
        if len(msg.missing_keys) > 0:
            print(f"Missing keys ({len(msg.missing_keys)}):")
            for k in msg.missing_keys:
                print(f"  - {k}")
        else:
            print(f"ALL Backbone keys matched perfectly!")

        # 2. 打印 Unexpected keys (预训练权重里有，但你当前模型不需要的)
        if len(msg.unexpected_keys) > 0:
            print(f"\nUnexpected keys ({len(msg.unexpected_keys)}):")
            for k in msg.unexpected_keys[:20]:  # 打印前20个防止刷屏
                print(f"  - {k}")
            if len(msg.unexpected_keys) > 20:
                print("  ... (and more)")
        print(f"==================================================\n")

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.backbone.patch_embed(x)
        pos_embed = self.backbone.interpolate_pos_encoding(x, self.backbone.pos_embed)
        x = x + pos_embed

        features = []
        Hp, Wp = self.img_size // self.patch_size, self.img_size // self.patch_size
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x.permute(0, 2, 1).reshape(B, self.embed_dim, Hp, Wp)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        return tuple(features)

    def forward(self, x):
        # 处理单通道医疗图像输入（复制为3通道以适配ViT）
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.forward_features(x)
        logits = self.decode_head(features)
        out = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return out