# train.py
import argparse
import collections
import pandas as pd
import torch
import numpy as np
import random
import wandb
import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning)
from timm.scheduler import CosineLRScheduler # 引入官方使用的调度器
from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.trainer.trainer import Trainer
import src.utils.losses as loss_module
import src.utils.metrics as metric_module
from src.utils.parse_config import ConfigParser
from src.utils.util import count_params
import src.models.cnn_based as cnn_models
import src.models.ViT_based as transformer_models


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_freezing(model, freeze_mode):
    """Applies weight freezing to the model based on the specified mode."""
    if freeze_mode == 'none' or freeze_mode is None:
        print("No layers frozen. Training all parameters.")
        return model

    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == 'encoder':
        if hasattr(model, 'encoder'):
            print("Freezing ENCODER weights...")
            for param in model.encoder.parameters():
                param.requires_grad = False

        elif hasattr(model, 'backbone'):
            print("Freezing BACKBONE weights...")
            for param in model.backbone.parameters():
                param.requires_grad = False

        else:
            print(
                f"Warning: freeze_mode is 'encoder' but model {type(model).__name__} has no 'encoder' attribute. Training all params.")

    return model


def get_usfm_layer_id(name, depth=12):
    """精确匹配官方的分层逻辑，计算每个参数属于第几层"""
    num_layers = depth + 2
    if "cls_token" in name or "pos_embed" in name or "mask_token" in name:
        return 0
    elif "patch_embed" in name:
        return 0
    elif "rel_pos_bias" in name:
        return num_layers - 1
    elif "blocks" in name:
        # 提取 blocks 的层级 index: 如 'backbone.blocks.5.attn...' -> 5
        layer_id = int(name.split("blocks.")[1].split(".")[0])
        return layer_id + 1
    else:
        # 剩下的（FPN, Decode Head, Aux Head 等等）属于最顶层
        return num_layers - 1


def build_official_usfm_optimizer(model, config_usfm):
    """构建自带 Layer Decay 和 Weight Decay 免除规则的 AdamW"""
    lr = config_usfm.BASE_LR
    weight_decay = config_usfm.WEIGHT_DECAY
    layer_decay = config_usfm.LAYER_DECAY

    depth = model.backbone.depth if hasattr(model, 'backbone') else 12
    num_layers = depth + 2

    # 算好每一层的缩放比例 (比如第 13 层是 1.0, 越往下越呈 0.65 指数衰减)
    scales = [layer_decay ** i for i in reversed(range(num_layers))]

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = get_usfm_layer_id(name, depth)
        scale = scales[layer_id]

        # 官方 Tricks：对于一维张量(如 LayerNorm 权重)和所有的 bias，不使用权重衰减
        if len(param.shape) == 1 or name.endswith(".bias"):
            wd = 0.0
        else:
            wd = weight_decay

        group_name = f"layer_{layer_id}_wd_{wd}"
        if group_name not in param_groups:
            # 👇 多加一个 "name": group_name，方便 debug 打印
            param_groups[group_name] = {"params": [], "lr": lr * scale, "weight_decay": wd, "name": group_name}
        param_groups[group_name]["params"].append(param)

    print(
        f"✨ [Official Tricks] 优化器 AdamW 已构建! Base_LR={lr}, Weight_Decay={weight_decay}, Layer_Decay={layer_decay}")
    return torch.optim.AdamW(list(param_groups.values()), eps=1e-8, betas=(0.9, 0.999))


def build_official_usfm_scheduler(optimizer, epochs, config_usfm):
    """构建包含 Warmup 的 CosineLRScheduler (总 Epoch 沿用你的配置)"""
    print(
        f"✨ [Official Tricks] 调度器 CosineLRScheduler 已构建! Warmup_Epochs={config_usfm.WARMUP_EPOCHS}, Min_LR={config_usfm.MIN_LR}")
    return CosineLRScheduler(
        optimizer,
        t_initial=epochs,  # 沿用你设置的 epochs，无需写死 400
        lr_min=config_usfm.MIN_LR,
        warmup_lr_init=5e-5,  # 官方 Warmup 起始 LR
        warmup_t=config_usfm.WARMUP_EPOCHS,
        t_in_epochs=True,
    )




# ======================================================================

def main(config):
    set_seed(config['system']['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer_config = config['trainer']
    data_config = config['data']

    for fold in range(1, trainer_config['k_folds'] + 1):
        print(f"\n{'=' * 20} FOLD {fold}/{trainer_config['k_folds']} {'=' * 20}")

        run_name = f"{config['name']}_fold{fold}"

        preparer = PrepareDataset()
        if data_config['force_prepare']:
            print("Forcing dataset preparation...")
            preparer.run(dataset_list=data_config['datasets'])

        all_dfs = []
        for name in data_config['datasets']:
            csv_path = preparer.data_dir / f"{name}.csv"
            if csv_path.exists():
                all_dfs.append(pd.read_csv(csv_path))
            else:
                raise FileNotFoundError(f"{csv_path} not found. Please run with --force_prepare first.")
        df = pd.concat(all_dfs, ignore_index=True)

        loader_args = {
            'batch_size': data_config['batch_size'],
            'num_workers': data_config['num_workers'],
            'target_size': data_config['target_size'],
            'use_pad': data_config.get('use_pad', True)
        }

        train_loader = BUSDataLoader(df, **loader_args, split=str(fold), is_test=False, augment=True)
        df_val = df[df['split'] == str(fold)].copy()
        val_loader = BUSDataLoader(df_val, **loader_args, split=str(fold), is_test=True)

        # ==================== 输入前数据可视化 (Sanity Check) ====================
        if fold == 1:
            print("\n[Debug] 正在生成输入数据可视化图片...")
            batch = next(iter(train_loader))
            images = batch['image']
            masks = batch['mask']

            num_samples = min(4, images.size(0))
            img_grid = torchvision.utils.make_grid(images[:num_samples], nrow=num_samples, normalize=False)
            mask_grid = torchvision.utils.make_grid(masks[:num_samples], nrow=num_samples, normalize=False)

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.title(f"Input Images (Shape: {images.shape}, Min: {images.min():.2f}, Max: {images.max():.2f})")
            plt.imshow(img_grid.permute(1, 2, 0).squeeze().numpy(), cmap='gray')
            plt.axis('off')

            plt.subplot(2, 1, 2)
            plt.title(f"Target Masks (Shape: {masks.shape}, Min: {masks.min():.2f}, Max: {masks.max():.2f})")
            plt.imshow(mask_grid.permute(1, 2, 0).squeeze().numpy(), cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            save_path = f"debug_input_batch_fold{fold}.png"
            plt.savefig(save_path, dpi=150)
            print(f"[Debug] 可视化图片已保存至: {save_path}\n")
        # ==============================================================================

        model_type = config['arch']['type']
        if hasattr(cnn_models, model_type):
            model = config.init_obj('arch', cnn_models)
        elif model_type in ["TransUnet", "SwinUnet", "MedT", "JEPA_UPerNet", "USFM_UPerNet"]:
            model = transformer_models.get_transformer_based_model(
                model_name=model_type,
                config=config.config,
                num_classes=1
            )
        else:
            raise ValueError(f"Model type '{model_type}' not found in cnn_based or ViT_based models.")

        if config.transfer_from and config.transfer_from.exists():
            print(f"\nLoading weights for transfer learning from: {config.transfer_from}")
            checkpoint = torch.load(config.transfer_from, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Weights loaded successfully.")

        freeze_mode = config['trainer'].get('freeze_mode', 'none')
        model = apply_freezing(model, freeze_mode)
        model = model.to(device)

        # ======================= 模型参数与预训练权重核对 =======================
        if fold == 1:
            print("\n" + "=" * 50)
            print("  [Sanity Check] 模型参数与预训练权重核对")
            print("=" * 50)

            print("\n🟢 1. 当前模型 (Model) 的参数示例 (前10层):")
            model_keys = list(model.state_dict().keys())
            for k in model_keys[:10]:
                print(f"  - {k}: {model.state_dict()[k].shape}")
            print(f"  ... (当前模型总计包含 {len(model_keys)} 个 Tensor)\n")

            pretrained_path = None
            if config['arch']['type'] == 'JEPA_UPerNet':
                pretrained_path = config.config.get('jepa_args', {}).get('PRETRAIN_CKPT')
            elif model_type == 'USFM_UPerNet':
                pretrained_path = config.config.get('usfm_args', {}).get('PRETRAIN_CKPT')
            elif config['arch']['type'] == 'SwinUnet':
                pretrained_path = config.config.get('swin_unet_args', {}).get('PRETRAIN_CKPT')

            if pretrained_path and Path(pretrained_path).exists():
                print(f"🔵 2. 读取预训练权重文件: {pretrained_path}")
                ckpt = torch.load(pretrained_path, map_location='cpu')

                if 'target_encoder' in ckpt:
                    raw_state_dict = ckpt['target_encoder']
                elif 'model' in ckpt:
                    raw_state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    raw_state_dict = ckpt['state_dict']
                else:
                    raw_state_dict = ckpt

                print("  预训练文件中的参数示例 (前10层):")
                ckpt_keys = list(raw_state_dict.keys())
                for k in ckpt_keys[:10]:
                    print(f"  - {k}: {raw_state_dict[k].shape}")
                print(f"  ... (预训练文件总计包含 {len(ckpt_keys)} 个 Tensor)\n")

                clean_model_keys = [k.split('.')[-1] for k in model_keys]
                clean_ckpt_keys = [k.split('.')[-1] for k in ckpt_keys]
                matched_count = sum(1 for k in clean_ckpt_keys if k in clean_model_keys)
                print(f"🟠 3. 粗略匹配评估 (仅比对后缀名):")
                print(f"  预训练文件中大约有 {matched_count} / {len(ckpt_keys)} 个参数层能在当前模型中找到对应结构。")
                print("  (注：详细的严格匹配缺失项请参考模型内部 load_from 的打印输出)")
            else:
                print(f"🔵 2. 未找到预训练权重，或当前模型无需预训练权重。")
            print("=" * 50 + "\n")
        # ======================= 插入代码结束 =======================

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

        if not config['wandb']['disable']:
            wandb.init(
                project=config['wandb']['project'],
                name=run_name,
                config=config.config,
                reinit=True
            )
            wandb.config.update({
                'trainable_params_M': round(trainable_params / 1e6, 2),
                'freeze_mode': freeze_mode
            }, allow_val_change=True)

        trainer_run_config = config.config.copy()
        trainer_run_config['checkpoint_name'] = f"{run_name}_best.pth"
        for key, value in config['trainer'].items():
            trainer_run_config[key] = value

        # ======================= 动态构建优化器与调度器 =======================
        is_official_usfm = (model_type == 'USFM_UPerNet' and hasattr(model, 'mode') and model.mode == 'official')

        if is_official_usfm:
            # 开启 Official 神装
            usfm_cfg = model.config.MODEL.USFM
            optimizer = build_official_usfm_optimizer(model, usfm_cfg)
            epochs = trainer_run_config['epochs']  # 直接读你的 800 (如果有早停依然生效)
            lr_scheduler = build_official_usfm_scheduler(optimizer, epochs, usfm_cfg)

            # 将必要变量注入给 trainer
            trainer_run_config['usfm_aux_weight'] = usfm_cfg.AUX_WEIGHT
            trainer_run_config['is_timm_scheduler'] = True  # 标记当前调度器属于 timm
        else:
            # 原本的统一初始化逻辑 (完全保持不变，兼容其它所有模型)
            trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = config.init_obj('optimizer', torch.optim, trainable_model_params)
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
            trainer_run_config['is_timm_scheduler'] = False

        criterion = config.init_obj('loss', loss_module)
        metrics = [getattr(metric_module, met) for met in config['metrics']]

        # ======================= 👇 新增：在此处插入打印代码 👇 =======================
        if fold == 1:
            print("\n" + "=" * 70)
            print("  🔍 [Sanity Check] 初始优化器参数分组 & 学习率分布")
            print("  (注: 此处显示的 LR 为 Warmup 结束后的最高目标学习率)")
            print("=" * 70)
            for i, group in enumerate(optimizer.param_groups):
                num_params = len(group['params'])
                # 如果有名字就拿名字，没有就叫 Standard_Group
                group_name = group.get('name', f'Standard_Group_{i}')
                lr = group['lr']
                wd = group['weight_decay']

                print(f"  - {group_name:20s} | 包含参数层数: {num_params:4d} | 目标LR: {lr:.4e} | WD: {wd:.4e}")
            print("=" * 70 + "\n")
        # ======================= 👆 插入结束 👆 =======================

        if not config['wandb']['disable']:
            wandb.watch(model, criterion, log="all", log_freq=100)

        # 启动 Trainer
        trainer = Trainer(
            model=model, criterion=criterion, metrics=metrics, optimizer=optimizer,
            config=trainer_run_config, device=device, train_loader=train_loader,
            val_loader=val_loader, lr_scheduler=lr_scheduler
        )

        if config.resume:
            trainer.resume_checkpoint(config.resume)

        try:
            trainer.train()
        finally:
            if not config['wandb']['disable']:
                wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Breast Ultrasound Segmentation')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--transfer-from', default=None, type=str,
                        help='path to checkpoint for transfer learning (loads weights only)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target kwargs')
    options = [
        CustomArgs(['--name'], str, 'name', {}),
        CustomArgs(['--model'], str, 'arch;type', {}),
        CustomArgs(['--datasets'], str, 'data;datasets', {'nargs': '+'}),
        CustomArgs(['--bs'], int, 'data;batch_size', {}),
        CustomArgs(['--size'], int, 'data;target_size', {}),
        CustomArgs(['--lr'], float, 'optimizer;args;lr', {}),
        CustomArgs(['--wd'], float, 'optimizer;args;weight_decay', {}),
        CustomArgs(['--loss'], str, 'loss;type', {}),
        CustomArgs(['--scheduler'], str, 'lr_scheduler;type', {}),
        CustomArgs(['--epochs'], int, 'trainer;epochs', {}),
        CustomArgs(['--patience'], int, 'trainer;early_stopping_patience', {}),
        CustomArgs(['--freeze-mode'], str, 'trainer;freeze_mode', {'choices': ['none', 'encoder'], 'default': 'none'}),
    ]

    config = ConfigParser.from_args(parser, options)
    main(config)