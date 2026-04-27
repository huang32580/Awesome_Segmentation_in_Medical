import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.IMG_SIZE = 224

_C.MODEL = CN()
_C.MODEL.TYPE = 'jepa_upernet'
_C.MODEL.NAME = 'jepa_vit_small_patch16_224'
_C.MODEL.PRETRAIN_CKPT = None

_C.MODEL.JEPA = CN()
_C.MODEL.JEPA.MODEL_NAME = 'vit_small'
_C.MODEL.JEPA.PATCH_SIZE = 16
_C.MODEL.JEPA.EMBED_DIM = 384
_C.MODEL.JEPA.OUT_INDICES = [3, 5, 7, 11]

_C.MODEL.USFM = CN()
_C.MODEL.USFM.MODEL_NAME = 'vit_base'
_C.MODEL.USFM.EMBED_DIM = 768
_C.MODEL.USFM.DEPTH = 12
_C.MODEL.USFM.NUM_HEADS = 12
_C.MODEL.USFM.PATCH_SIZE = 16
_C.MODEL.USFM.OUT_INDICES = [3, 5, 7, 11]
_C.MODEL.USFM.USE_REL_POS_BIAS = False

# ================= 新增 USFM 官方模式控制 (替换掉上一版的 BACKBONE_LR_MULT) =================
_C.MODEL.USFM.MODE = 'local'           # 'local' 或 'official'
_C.MODEL.USFM.AUX_WEIGHT = 0.4         # 辅助 Loss 权重

# 官方训练超参神装 (仅在 MODE='official' 时生效)
_C.MODEL.USFM.BASE_LR = 3e-4           # 官方基础学习率
_C.MODEL.USFM.MIN_LR = 0.0             # 学习率降到最小
_C.MODEL.USFM.WEIGHT_DECAY = 0.05      # 针对 ViT 的重度权重衰减
_C.MODEL.USFM.LAYER_DECAY = 0.65       # 官方精密的 0.65 逐层指数衰减率
_C.MODEL.USFM.WARMUP_EPOCHS = 20       # 20 个 Epoch 的预热机制
# =========================================================================================

_C.MODEL.UPERNET = CN()
_C.MODEL.UPERNET.POOL_SCALES = [1, 2, 3, 6]

def get_config(cfg_file=None):
    config = _C.clone()
    if cfg_file:
        config.defrost()
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        config.merge_from_file(cfg_file)
        config.freeze()
    return config