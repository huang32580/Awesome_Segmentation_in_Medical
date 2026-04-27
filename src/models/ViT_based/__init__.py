import argparse
import numpy as np
import torch
from pathlib import Path

# 导入各模型的实现类
from .transUnet.transunet import TransUnet
from .swinUnet.vision_transformer import SwinUnet
from .swinUnet.config import get_config as get_swin_config
from .medicalT.axialnet import MedT
from .jepa_upernet.jepa_upernet import JEPA_UPerNet
from .jepa_upernet.usfm_upernet import USFM_UPerNet
from .jepa_upernet.config import get_config as get_jepa_config


def _prepare_yacs_config(model_name, config_json, section_key, img_size, project_root):
    """
    内部辅助函数：统一处理 JEPA 和 USFM 的配置合并逻辑。
    实现“单一事实来源”，让 JSON 里的 PRETRAIN_CKPT 拥有绝对优先级。
    """
    args_section = config_json.get(section_key, {})
    relative_yaml_path = args_section.get('cfg')

    if not relative_yaml_path:
        raise ValueError(f"模型 {model_name} 需要在 config.json 的 '{section_key}' 中指定 'cfg' 路径")

    # 1. 定位并加载 YAML 配置文件
    absolute_yaml_path = project_root / relative_yaml_path.lstrip('./')
    if not absolute_yaml_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {absolute_yaml_path}")

    # 获取 YACS 默认配置并加载 YAML
    yacs_config = get_jepa_config(str(absolute_yaml_path))
    yacs_config.defrost()

    # 2. 参数强制同步：用 JSON (主控台) 的参数覆盖 YAML
    if img_size:
        yacs_config.DATA.IMG_SIZE = img_size

    # 3. 权重路径唯一化：优先从 JSON 对应的 args 分支读取
    # 如果 JSON 里设置为 null 或 "none"，则明确不加载权重
    pretrained_path_str = args_section.get('PRETRAIN_CKPT')
    if pretrained_path_str and str(pretrained_path_str).lower() != 'none':
        absolute_ckpt_path = project_root / str(pretrained_path_str).lstrip('./')
        yacs_config.MODEL.PRETRAIN_CKPT = str(absolute_ckpt_path)
    else:
        yacs_config.MODEL.PRETRAIN_CKPT = "none"

    yacs_config.freeze()
    return yacs_config


def get_transformer_based_model(model_name: str, config: dict, num_classes: int = 1):
    """
    工厂函数：根据配置实例化对应的 Transformer 分割模型。
    """
    data_config = config.get('data', {})
    img_size = data_config.get('target_size')
    in_channels = 1  # 基础输入通道

    # 定位项目根目录 (src/models/ViT_based/__init__.py 的上四级)
    project_root = Path(__file__).parent.parent.parent.parent

    # ---------------------------------------------------------
    # 1. JEPA_UPerNet 分支
    # ---------------------------------------------------------
    if model_name == "JEPA_UPerNet":
        jepa_config_obj = _prepare_yacs_config(
            model_name, config, 'jepa_args', img_size, project_root
        )
        model = JEPA_UPerNet(config=jepa_config_obj, num_classes=num_classes)
        model.load_from()  # 内部会根据 PRETRAIN_CKPT 是否为 "none" 决定是否加载
        return model

    # ---------------------------------------------------------
    # 2. USFM_UPerNet 分支
    # ---------------------------------------------------------
    elif model_name == "USFM_UPerNet":
        usfm_config_obj = _prepare_yacs_config(
            model_name, config, 'usfm_args', img_size, project_root
        )
        model = USFM_UPerNet(config=usfm_config_obj, num_classes=num_classes)
        model.load_from()  # 内部包含针对 USFM 键名的清洗逻辑
        return model

    # ---------------------------------------------------------
    # 3. SwinUnet 分支
    # ---------------------------------------------------------
    elif model_name == "SwinUnet":
        swin_section = config.get('swin_unet_args', {})
        relative_yaml_path = swin_section.get('cfg')
        if not relative_yaml_path:
            raise ValueError("SwinUnet 需要在 config.json 的 'swin_unet_args' 中指定 'cfg' 路径")

        absolute_yaml_path = project_root / relative_yaml_path.lstrip('./')

        # 封装 Swin 专用的 Namespace 参数
        swin_args = argparse.Namespace(
            cfg=str(absolute_yaml_path),
            batch_size=data_config.get('batch_size'),
            zip=swin_section.get('zip', False),
            cache_mode=swin_section.get('cache_mode', 'part'),
            resume=swin_section.get('resume'),
            opts=swin_section.get('opts'),
            accumulation_steps=swin_section.get('accumulation-steps'),
            use_checkpoint=swin_section.get('use-checkpoint'),
            amp_opt_level=None, tag='default', eval=False, throughput=False
        )

        swin_config_obj = get_swin_config(swin_args)
        swin_config_obj.defrost()
        swin_config_obj.MODEL.SWIN.IN_CHANS = in_channels

        # 同步预训练权重
        pretrained_path_str = swin_section.get('PRETRAIN_CKPT')
        if pretrained_path_str:
            swin_config_obj.MODEL.PRETRAIN_CKPT = str(project_root / pretrained_path_str.lstrip('./'))

        swin_config_obj.freeze()
        model = SwinUnet(config=swin_config_obj, img_size=img_size, num_classes=num_classes)

        if swin_config_obj.MODEL.PRETRAIN_CKPT:
            model.load_from(swin_config_obj)
        return model

    # ---------------------------------------------------------
    # 4. TransUnet 分支
    # ---------------------------------------------------------
    elif model_name == "TransUnet":
        model = TransUnet(img_size=img_size, img_ch=in_channels, output_ch=num_classes)
        vit_args = config.get('vit_seg_args', {})
        if vit_args.get('vit_patches_path'):
            model.load_from(weights_npz_path=vit_args['vit_patches_path'])
        return model

    # ---------------------------------------------------------
    # 5. MedT 分支
    # ---------------------------------------------------------
    elif model_name == "MedT":
        model = MedT(img_size=img_size, imgchan=in_channels, num_classes=num_classes)
        return model

    else:
        raise ValueError(f"无法识别的 Transformer 模型: '{model_name}'")