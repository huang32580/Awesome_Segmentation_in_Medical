import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from functools import partial

# 导入你上传的 USFM 版 VisionTransformer
from .vision_transformer_usfm import VisionTransformer as USFM_ViT
from .jepa_upernet import ConvModule, SimpleUPerHead

# 导入你刚刚从官方搬运过来的 SegViT 解码器
from .atm_head import ATMHead


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


class USFM(nn.Module):
    """
    统一的 USFM 分割模型入口，支持动态切换 UPerHead 与 SegViT(ATMHead)
    """

    def __init__(self, config, num_classes=1):
        super().__init__()

        self.usfm_args = config.get('usfm_args', {})
        self.mode = self.usfm_args.get('mode', 'local')
        self.decoder_type = self.usfm_args.get('decoder_type', None)
        if self.decoder_type is None:
            raise ValueError(
                "\n[ERROR] USFM 模型配置错误！\n"
                "必须在 'usfm_args' 中明确指定 'decoder_type'。\n"
                "可选值: 'UPerHead' (对应 UPerNet) 或 'SegViT' (对应 ATMHead)。"
            )

        self.patch_size = self.usfm_args.get('patch_size', 16)
        self.embed_dim = self.usfm_args.get('embed_dim', 768)
        self.out_indices = self.usfm_args.get('out_indices', [3, 5, 7, 11])
        self.img_size = config.get('data', {}).get('target_size', 224)

        depth = self.usfm_args.get('depth', 12)
        num_heads = self.usfm_args.get('num_heads', 12)
        drop_path_rate = self.usfm_args.get('drop_path_rate', 0.1)

        # 1. 实例化通用的 USFM Backbone
        self.backbone = USFM_ViT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            use_abs_pos_emb=False,
            use_shared_rel_pos_bias=True,
            init_values=0.1,
            use_mean_pooling=False,
            drop_path_rate=drop_path_rate
        )

        # 2. 动态构建解码器
        if self.decoder_type == 'UPerHead':
            print(f"🚀 [USFM] 已选择 UPerHead 解码器 (Mode: {self.mode})")
            self.pool_scales = [1, 2, 3, 6]
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(self.embed_dim), nn.GELU(),
                nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.decode_head = SimpleUPerHead(
                in_channels_list=[self.embed_dim] * 4,
                channels=self.embed_dim,
                pool_scales=self.pool_scales,
                num_classes=num_classes
            )
            if self.mode == 'official':
                self.aux_head = FCNHead(self.embed_dim, 256, num_classes)

        elif self.decoder_type == 'SegViT':
            print(f"🚀 [USFM] 已选择 SegViT 解码器 (Mode: {self.mode})")
            self.decode_head = ATMHead(
                img_size=self.img_size,
                in_channels=[self.embed_dim] * 4,
                embed_dims=self.embed_dim,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        self._init_weights()

    def _init_weights(self):
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
        pretrained_path = self.usfm_args.get('PRETRAIN_CKPT', None)
        if not pretrained_path or str(pretrained_path).lower() == 'none':
            print("WARNING: No USFM pretrain checkpoint. Training FROM SCRATCH.")
            return

        if not os.path.exists(pretrained_path):
            print(f"Error: Path {pretrained_path} not found.")
            return

        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

        new_state_dict = {}
        for k, v in state_dict.items():
            if 'predictor' in k or 'mask_token' in k or 'head' in k:
                continue
            clean_k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '')
            new_state_dict[clean_k] = v

        msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"\nUSFM Pretrained Weights Loaded. Missing Keys: {len(msg.missing_keys)}")

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

        return tuple(features)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. 提取 Backbone 纯特征 [B, C, H, W] 的列表
        features = self.forward_features(x)

        # 2. 动态前传 Decoder 与输出分离
        if self.decoder_type == 'SegViT':
            # 官方 ATMHead 通常设计为直接吞入 ViT 的多层特征输出，内部完成插值
            out_dict = self.decode_head(features)

            # =======================================================
            # 🚀 核心修复：Double Sigmoid Bug (Logit Inverse)
            # 因为 Awesome 框架的 metrics 和 loss 默认会对输出求 sigmoid。
            # 而 ATMHead 已经在内部算出了 [0, 1] 之间的严格概率。
            # 这里将其逆变换回 Logits！
            # =======================================================
            prob = out_dict["pred"]
            # 截断以防止出现 log(0) 导致 loss 或评估变成 NaN
            prob = torch.clamp(prob, 1e-7, 1.0 - 1e-7)
            fake_logits = torch.log(prob / (1.0 - prob))

            # 用 fake_logits 替换掉原有的概率图
            out_dict["pred"] = fake_logits

            return out_dict

        elif self.decoder_type == 'UPerHead':
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            head_inputs = [ops[i](features[i]) for i in range(len(features))]
            logits = self.decode_head(head_inputs)

            # 插值还原至原图大小
            out_main = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

            # 辅助头处理 (只在训练+官方模式开启)
            if self.training and self.mode == 'official':
                aux_logits = self.aux_head(features[2])
                out_aux = F.interpolate(aux_logits, size=(self.img_size, self.img_size), mode='bilinear',
                                        align_corners=False)
                return out_main, out_aux

            return out_main