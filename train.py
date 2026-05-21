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


def build_official_usfm_optimizer(model, usfm_args):
    lr = usfm_args.get('base_lr', 1e-4)
    weight_decay = usfm_args.get('weight_decay', 0.05)
    layer_decay = usfm_args.get('layer_decay', 0.65)

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


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def summarize_optimizer(optimizer):
    group_lrs = [group.get('lr', 0.0) for group in optimizer.param_groups]
    group_wds = [group.get('weight_decay', 0.0) for group in optimizer.param_groups]
    group_params = [sum(p.numel() for p in group.get('params', [])) for group in optimizer.param_groups]
    return {
        'groups': len(optimizer.param_groups),
        'lr_min': min(group_lrs) if group_lrs else None,
        'lr_max': max(group_lrs) if group_lrs else None,
        'weight_decays': sorted(set(group_wds)),
        'group_param_min': min(group_params) if group_params else 0,
        'group_param_max': max(group_params) if group_params else 0,
    }


def print_training_debug(model, model_type, usfm_args, freeze_mode, optimizer, lr_scheduler, trainer_run_config, criterion):
    print("\n" + "=" * 100)
    print("[DEBUG] Training setup before first optimizer step")
    print(f"[DEBUG] model_type={model_type}")
    print(f"[DEBUG] freeze_mode={freeze_mode}")
    if model_type == 'USFM':
        print(f"[DEBUG] usfm_mode={usfm_args.get('mode', 'local')}")
        print(f"[DEBUG] decoder_type={usfm_args.get('decoder_type')}")
        print(f"[DEBUG] pretrain_ckpt={usfm_args.get('PRETRAIN_CKPT')}")
        print(f"[DEBUG] out_indices={getattr(model, 'out_indices', None)}")
        print(f"[DEBUG] has_aux_head={hasattr(model, 'aux_head')}")
        print(f"[DEBUG] usfm_aux_weight={trainer_run_config.get('usfm_aux_weight')}")
        print(f"[DEBUG] is_timm_scheduler={trainer_run_config.get('is_timm_scheduler')}")
        pretrain_debug = getattr(model, 'pretrain_debug', None)
        if pretrain_debug is None:
            print("[DEBUG] pretrain_debug=missing (model did not expose load statistics)")
        else:
            print(
                "[DEBUG] pretrain_loaded={loaded}, matched={matched}, missing={missing}, "
                "unexpected={unexpected}, shape_mismatch={shape_mismatch}".format(**pretrain_debug)
            )
            for item in pretrain_debug.get('shape_mismatch_examples', []):
                print(f"[DEBUG] pretrain_shape_mismatch_example={item}")

    total, trainable = count_params(model)
    print(f"[DEBUG] params_total={total:,}")
    print(f"[DEBUG] params_trainable={trainable:,}")
    print(f"[DEBUG] params_frozen={total - trainable:,}")
    if hasattr(model, 'backbone'):
        b_total, b_trainable = count_params(model.backbone)
        print(f"[DEBUG] backbone_total={b_total:,}")
        print(f"[DEBUG] backbone_trainable={b_trainable:,}")
        print(f"[DEBUG] backbone_frozen={b_total - b_trainable:,}")
        print(f"[DEBUG] backbone_is_frozen={b_total > 0 and b_trainable == 0}")

    opt_summary = summarize_optimizer(optimizer)
    print(f"[DEBUG] optimizer={optimizer.__class__.__name__}")
    print(f"[DEBUG] optimizer_groups={opt_summary['groups']}")
    print(f"[DEBUG] optimizer_lr_min={opt_summary['lr_min']}")
    print(f"[DEBUG] optimizer_lr_max={opt_summary['lr_max']}")
    print(f"[DEBUG] optimizer_weight_decays={opt_summary['weight_decays']}")
    print(f"[DEBUG] optimizer_group_param_min={opt_summary['group_param_min']:,}")
    print(f"[DEBUG] optimizer_group_param_max={opt_summary['group_param_max']:,}")
    print(f"[DEBUG] scheduler={lr_scheduler.__class__.__name__ if lr_scheduler is not None else None}")
    print(f"[DEBUG] criterion={criterion.__class__.__name__}")
    print("=" * 100 + "\n")


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

        # 🌟 修复划分验证集的问题核心：防范 Pandas 将 split 推断为数字，造成验证集切分为空
        df['split'] = df['split'].astype(str).str.strip()


        loader_args = {
            'batch_size': data_config['batch_size'],
            'num_workers': data_config['num_workers'],
            'target_size': data_config['target_size'],
            'use_pad': data_config.get('use_pad', True)
        }

        train_loader = BUSDataLoader(df, **loader_args, split=str(fold), is_test=False, augment=True)
        # 遵循 v1 显式 copy 提取的风格，更加稳定
        df_val = df[df['split'] == str(fold)].copy()
        val_loader = BUSDataLoader(df_val, **loader_args, split=str(fold), is_test=True)

        model_type = config['arch']['type']
        if hasattr(cnn_models, model_type):
            model = config.init_obj('arch', cnn_models)
        elif model_type in ["TransUnet", "SwinUnet", "MedT", "JEPA_UPerNet", "USFM"]:
            model = transformer_models.get_transformer_based_model(
                model_name=model_type,
                config=config.config,
                num_classes=1
            )
        else:
            raise ValueError(f"Model type '{model_type}' not found.")

        if config.transfer_from and config.transfer_from.exists():
            checkpoint = torch.load(config.transfer_from, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])

        freeze_mode = trainer_config.get('freeze_mode', config.config.get('freeze_mode', 'none'))
        if 'freeze_mode' in config.config and 'freeze_mode' not in trainer_config:
            print(
                "[DEBUG] Found top-level freeze_mode. It will be honored for this run, "
                "but prefer trainer.freeze_mode in config files."
            )
        model = apply_freezing(model, freeze_mode)
        model = model.to(device)

        if not config['wandb']['disable']:
            wandb.init(project=config['wandb']['project'], name=run_name, config=config.config, reinit=True)

        trainer_run_config = config.config.copy()
        trainer_run_config['checkpoint_name'] = f"{run_name}_best.pth"
        for key, value in config['trainer'].items():
            trainer_run_config[key] = value

        usfm_args = config.config.get('usfm_args', {})
        is_official_usfm = (model_type == 'USFM' and usfm_args.get('mode', 'local') == 'official')

        if is_official_usfm:
            print("\n🚀 [状态] 检测到 USFM 官方模式，挂载定制化优化器/调度器...")
            print(
                f"[DEBUG] Official optimizer settings: base_lr={usfm_args.get('base_lr', 1e-4)}, "
                f"weight_decay={usfm_args.get('weight_decay', 0.05)}, "
                f"layer_decay={usfm_args.get('layer_decay', 0.65)}"
            )
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
        print_training_debug(
            model=model,
            model_type=model_type,
            usfm_args=usfm_args,
            freeze_mode=freeze_mode,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            trainer_run_config=trainer_run_config,
            criterion=criterion
        )

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
