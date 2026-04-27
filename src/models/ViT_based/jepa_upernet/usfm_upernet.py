import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from functools import partial

# 导入你上传的 USFM 版 VisionTransformer
from .vision_transformer_usfm import VisionTransformer as USFM_ViT
from .jepa_upernet import ConvModule, SimpleUPerHead


# ================= 新增：辅助分类头 FCNHead =================
class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================================

class USFM_UPerNet(nn.Module):
    def __init__(self, config, num_classes=1):
        super().__init__()
        self.config = config

        self.patch_size = config.MODEL.USFM.PATCH_SIZE
        self.embed_dim = config.MODEL.USFM.EMBED_DIM
        self.out_indices = config.MODEL.USFM.OUT_INDICES
        self.img_size = config.DATA.IMG_SIZE
        self.pool_scales = config.MODEL.UPERNET.POOL_SCALES

        # 获取模式
        self.mode = getattr(config.MODEL.USFM, 'MODE', 'local')

        # 1. 实例化 USFM 专用 Backbone
        self.backbone = USFM_ViT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            use_abs_pos_emb=False,
            use_shared_rel_pos_bias=True,
            init_values=0.1,
            use_mean_pooling=False
        )

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

        # ================= 新增：如果是官方模式，则增加一个辅助头 =================
        if self.mode == 'official':
            # 辅助头一般接在倒数第二层特征上（即features的index=2的位置）
            self.aux_head = FCNHead(self.embed_dim, self.embed_dim, num_classes)
        # ====================================================================

        self._init_weights()

    def _init_weights(self):
        """符合学术规范的初始化"""
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
        """针对 USFM 权重的加载与硬核 Debug 逻辑"""
        pretrained_path = self.config.MODEL.PRETRAIN_CKPT
        if not pretrained_path or str(pretrained_path).lower() == 'none':
            print("WARNING: No USFM pretrain checkpoint. Training FROM SCRATCH.")
            return

        if not os.path.exists(pretrained_path):
            print(f"Error: Path {pretrained_path} not found.")
            return

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

        print("\n" + "▼" * 60)
        print("🔍 [DEBUG] 预训练权重 (Checkpoint) 中的所有参数:")
        for k, v in state_dict.items():
            print(f"  [CKPT] {k}: {v.shape}")

        print("\n" + "▼" * 60)
        print("🔍 [DEBUG] 当前 USFM Backbone 需要的所有参数:")
        for k, v in self.backbone.state_dict().items():
            print(f"  [Model] {k}: {v.shape}")
        print("▲" * 60 + "\n")

        new_state_dict = {}
        for k, v in state_dict.items():
            if 'predictor' in k or 'mask_token' in k or 'head' in k:
                continue
            clean_k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '')
            new_state_dict[clean_k] = v

        msg = self.backbone.load_state_dict(new_state_dict, strict=False)

        print(f"\n" + "=" * 60)
        print(f"USFM Pretrained Weights Loading Report")
        print(f"Path: {pretrained_path}")
        print(f"Matched Keys: {len(new_state_dict) - len(msg.missing_keys)}")

        if len(msg.missing_keys) > 0:
            print("\n❌ 严格缺失的 Keys (Model 有，但 CKPT 里没给，将保持随机初始化):")
            for k in msg.missing_keys:
                print(f"  - {k}")

        if len(msg.unexpected_keys) > 0:
            print("\n⚠️ 多余的 Keys (CKPT 给出了，但 Model 没地方放，被丢弃):")
            for k in msg.unexpected_keys:
                print(f"  - {k}")
        print("=" * 60 + "\n")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        rel_pos_bias = self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None

        features = []
        Hp, Wp = self.img_size // self.patch_size, self.img_size // self.patch_size

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, Hp, Wp)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.forward_features(x)
        logits = self.decode_head(features)
        out_main = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # ================= 新增：仅在 official 模式下的训练阶段返回元组 =================
        if self.training and self.mode == 'official':
            # features[2] 代表第 3 个输出层（往往是深层特征，适合接 Aux Head）
            aux_logits = self.aux_head(features[2])
            out_aux = F.interpolate(aux_logits, size=(self.img_size, self.img_size), mode='bilinear',
                                    align_corners=False)
            return out_main, out_aux
        # ==========================================================================

        return out_main