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
from timm.scheduler import CosineLRScheduler
from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.trainer.trainer import Trainer
import src.utils.losses as loss_module
import src.utils.metrics as metric_module
from src.utils.parse_config import ConfigParser
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
    if freeze_mode == 'none' or freeze_mode is None:
        return model
    for param in model.parameters():
        param.requires_grad = True
    if freeze_mode == 'encoder':
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters(): param.requires_grad = False
        elif hasattr(model, 'backbone'):
            for param in model.backbone.parameters(): param.requires_grad = False
    return model


def get_usfm_layer_id(name, depth=12):
    num_layers = depth + 2
    if "cls_token" in name or "pos_embed" in name or "mask_token" in name or "patch_embed" in name:
        return 0
    elif "rel_pos_bias" in name:
        return num_layers - 1
    elif "blocks" in name:
        layer_id = int(name.split("blocks.")[1].split(".")[0])
        return layer_id + 1
    else:
        return num_layers - 1


# 改为接收字典
def build_official_usfm_optimizer(model, usfm_args):
    lr = usfm_args.get('base_lr', 1e-4)
    weight_decay = usfm_args.get('weight_decay', 0.05)
    layer_decay = usfm_args.get('layer_decay', 0.65)

    # 🚀 直接从配置字典中读取 depth，默认值为 12
    depth = usfm_args.get('depth', 12)
    num_layers = depth + 2

    scales = [layer_decay ** i for i in reversed(range(num_layers))]
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        layer_id = get_usfm_layer_id(name, depth)
        scale = scales[layer_id]

        if len(param.shape) == 1 or name.endswith(".bias"):
            wd = 0.0
        else:
            wd = weight_decay

        group_name = f"layer_{layer_id}_wd_{wd}"
        if group_name not in param_groups:
            param_groups[group_name] = {"params": [], "lr": lr * scale, "weight_decay": wd, "name": group_name}
        param_groups[group_name]["params"].append(param)

    print(f"✨ [Official Tricks] 构建官方 AdamW! Base_LR={lr}, WD={weight_decay}, Decay={layer_decay}")
    return torch.optim.AdamW(list(param_groups.values()), eps=1e-8, betas=(0.9, 0.999))


def build_official_usfm_scheduler(optimizer, epochs, usfm_args):
    warmup_ep = usfm_args.get('warmup_epochs', 20)
    min_lr = usfm_args.get('min_lr', 0.0)
    warmup_lr = usfm_args.get('warmup_lr', 5e-5)
    print(f"✨ [Official Tricks] 构建官方 CosineLRScheduler! Warmup={warmup_ep}, Min_LR={min_lr}")
    return CosineLRScheduler(
        optimizer, t_initial=epochs, lr_min=min_lr,
        warmup_lr_init=warmup_lr, warmup_t=warmup_ep, t_in_epochs=True,
    )


def main(config):
    set_seed(config['system']['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer_config = config['trainer']
    data_config = config['data']

    for fold in range(1, trainer_config['k_folds'] + 1):
        print(f"\n{'=' * 20} FOLD {fold}/{trainer_config['k_folds']} {'=' * 20}")
        run_name = f"{config['name']}_fold{fold}"

        preparer = PrepareDataset()
        if data_config['force_prepare']: preparer.run(dataset_list=data_config['datasets'])

        all_dfs = [pd.read_csv(preparer.data_dir / f"{name}.csv") for name in data_config['datasets']]
        df = pd.concat(all_dfs, ignore_index=True)

        loader_args = {
            'batch_size': data_config['batch_size'],
            'num_workers': data_config['num_workers'],
            'target_size': data_config['target_size'],
            'use_pad': data_config.get('use_pad', True)
        }

        train_loader = BUSDataLoader(df, **loader_args, split=str(fold), is_test=False, augment=True)
        val_loader = BUSDataLoader(df[df['split'] == str(fold)].copy(), **loader_args, split=str(fold), is_test=True)

        model_type = config['arch']['type']
        if hasattr(cnn_models, model_type):
            model = config.init_obj('arch', cnn_models)
        elif model_type in ["TransUnet", "SwinUnet", "MedT", "JEPA_UPerNet", "USFM_SegmentationModel","USFM_UPerNet"]:
            # 注意：这里的 config 传入的是 config.config 字典本身
            model = transformer_models.get_transformer_based_model(
                model_name=model_type,
                config=config.config,
                num_classes=1
            )
        else:
            raise ValueError(f"Model type '{model_type}' not found.")

        if config.transfer_from and config.transfer_from.exists():
            checkpoint = torch.load(config.transfer_from, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])

        model = apply_freezing(model, trainer_config.get('freeze_mode', 'none'))
        model = model.to(device)

        if not config['wandb']['disable']:
            wandb.init(project=config['wandb']['project'], name=run_name, config=config.config, reinit=True)

        trainer_run_config = config.config.copy()
        trainer_run_config['checkpoint_name'] = f"{run_name}_best.pth"
        for key, value in config['trainer'].items():
            trainer_run_config[key] = value

        # ======================= 动态构建优化器与调度器 =======================
        usfm_args = config.config.get('usfm_args', {})
        is_official_usfm = (model_type in ['USFM_UPerNet', 'USFM_SegmentationModel'] and usfm_args.get('mode',
                                                                                                       'local') == 'official')

        if is_official_usfm:
            print("\n🚀 [状态] 检测到 USFM 官方模式，挂载定制化优化器/调度器...")
            optimizer = build_official_usfm_optimizer(model, usfm_args)
            lr_scheduler = build_official_usfm_scheduler(optimizer, trainer_run_config['epochs'], usfm_args)

            trainer_run_config['usfm_aux_weight'] = usfm_args.get('aux_weight', 0.4)
            trainer_run_config['is_timm_scheduler'] = True
        else:
            print("\n🚀 [状态] 使用 Awesome 原生模式 (标准 Optimizer & Scheduler)...")
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

            trainer_run_config['usfm_aux_weight'] = 0.0
            trainer_run_config['is_timm_scheduler'] = False

        criterion = config.init_obj('loss', loss_module)
        metrics = [getattr(metric_module, met) for met in config['metrics']]

        trainer = Trainer(
            model=model, criterion=criterion, metrics=metrics, optimizer=optimizer,
            config=trainer_run_config, device=device, train_loader=train_loader,
            val_loader=val_loader, lr_scheduler=lr_scheduler
        )

        if config.resume: trainer.resume_checkpoint(config.resume)

        try:
            trainer.train()
        finally:
            if not config['wandb']['disable']: wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BUS Segmentation')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str)
    parser.add_argument('-d', '--device', default=None, type=str)
    parser.add_argument('--transfer-from', default=None, type=str)

    options = [
        collections.namedtuple('CustomArgs', 'flags type target kwargs')(['--name'], str, 'name', {})
    ]
    config = ConfigParser.from_args(parser, options)
    main(config)