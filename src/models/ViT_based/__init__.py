import os
import numpy as np
import torch
import warnings

# 忽略因为版本问题产生的一些无关紧要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================================
# 导入各个 Transformer 模型的依赖
# =========================================================================
from .transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from .transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .swinUnet.vision_transformer import SwinUnet as Swin_Unet
from .swinUnet.config import get_config
from .medicalT.axialnet import MedT
from .jepa_upernet.jepa_upernet import JEPA_UPerNet

# 🚀 重点：这里改为导入我们重构好的 USFM_SegmentationModel
from .jepa_upernet.usfm_upernet import USFM


def _prepare_yacs_config(config_dict, section_key):
    """
    辅助函数：为 JEPA_UPerNet 等仍依赖 YAML/YACS 文件的旧模型解析配置。
    (USFM 现已弃用此函数，直接读取 json dict)
    """
    from yacs.config import CfgNode as CN
    import yaml

    args = config_dict.get(section_key, {})
    cfg_path = args.get('cfg', None)

    if not cfg_path:
        raise ValueError(f"模型需要在 config.json 的 '{section_key}' 中指定 'cfg' 路径")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件未找到: {cfg_path}")

    with open(cfg_path, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 将 yaml 转为 yacs CfgNode
    cfg_node = CN(yaml_cfg)

    # 将 json 中覆盖的参数更新进去 (例如 PRETRAIN_CKPT)
    for k, v in args.items():
        if k != 'cfg':
            cfg_node[k] = v

    return cfg_node


def get_transformer_based_model(model_name, config, num_classes=1):
    """
    根据模型名称动态构建基于 Transformer 的分割模型。
    注意：这里的 config 传入的是 config.json 解析后的完整原始字典。
    """
    img_size = config['data']['target_size']

    # 1. TransUnet
    if model_name == 'TransUnet':
        vit_name = 'R50-ViT-B_16'
        vit_patches_size = 16
        vit_configs = CONFIGS_ViT_seg[vit_name]
        vit_configs.n_classes = num_classes
        vit_configs.split = 'train'
        vit_configs.n_skip = 3
        vit_configs.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

        model = ViT_seg(vit_configs, img_size=img_size, num_classes=num_classes)

        vit_seg_args = config.get('vit_seg_args', {})
        pretrain_path = vit_seg_args.get('vit_patches_path', None)
        if pretrain_path and os.path.exists(pretrain_path):
            model.load_from(weights=np.load(pretrain_path))
        return model

    # 2. SwinUnet
    elif model_name == 'SwinUnet':
        swin_args = config.get('swin_unet_args', {})
        cfg_path = swin_args.get('cfg', None)
        if not cfg_path:
            raise ValueError("SwinUnet 需要在 'swin_unet_args' 中指定 'cfg' 路径")

        import argparse
        args = argparse.Namespace(cfg=cfg_path, opts=None, zip=False, cache_mode='part', resume=None,
                                  accumulation_steps=None, use_checkpoint=False, amp_opt_level='O1',
                                  tag=None, eval=False, throughput=False)
        swin_config = get_config(args)

        model = Swin_Unet(config=swin_config, img_size=img_size, num_classes=num_classes)

        pretrain_path = swin_args.get('PRETRAIN_CKPT', None)
        if pretrain_path and os.path.exists(pretrain_path):
            model.load_from(swin_config)
        return model

    # 3. MedT
    elif model_name == 'MedT':
        model = MedT(img_size=img_size, num_classes=num_classes)
        return model

    # 4. JEPA_UPerNet
    elif model_name == 'JEPA_UPerNet':
        jepa_config_obj = _prepare_yacs_config(config, 'jepa_args')
        model = JEPA_UPerNet(config=jepa_config_obj, num_classes=num_classes)
        model.load_from()
        return model

    # =========================================================================
    # 5. 🚀 USFM 分支：完全弃用 yaml 和 yacs，直接传递 config 字典！
    # =========================================================================
    elif model_name == 'USFM':
        model = USFM(config=config, num_classes=num_classes)
        model.load_from()
        return model

    else:
        raise NotImplementedError(f"Model {model_name} is not supported yet.")