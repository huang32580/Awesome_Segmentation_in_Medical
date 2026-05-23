import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from torchvision.utils import make_grid

# Assume these utils and metrics are in the correct path
from src.utils.metrics import pixel_accuracy, dice_score, hd95_batch, iou_score
from src.utils.util import MetricTracker


class Trainer:
    def __init__(self, model, criterion, metrics, optimizer, config, device,
                 train_loader, val_loader=None, lr_scheduler=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        # 🚀 健壮性改进：显式定义输出属性
        self.arch_type = config.get('arch', {}).get('type', '')
        self.usfm_args = config.get('usfm_args', {})
        self.decoder_type = self.usfm_args.get('decoder_type', '')
        self.is_usfm_official = self.arch_type == 'USFM' and self.usfm_args.get('mode', 'local') == 'official'

        # 逻辑：只有 SegViT (ATMHead) 出来的 "pred" 是概率图，其他 (UPerHead, CNN) 都是 Logits
        if self.arch_type == 'USFM' and self.decoder_type == 'SegViT' and not self.is_usfm_official:
            self.output_is_prob = True
        else:
            self.output_is_prob = False

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.metric_fns = metrics

        train_metric_keys = ['loss', 'iou_score']
        self.train_metrics = MetricTracker(*train_metric_keys)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns])

        self.start_epoch = 1
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_name = config.get('checkpoint_name', 'best_model.pth')

        # 🚀 替换为平均排名策略所需的变量
        self.metric_history = []
        self.best_epoch_info = {}

        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.epochs_without_improvement = 0

        # 🌟 修复底层验证集空判问题
        self.fixed_val_batch = next(iter(self.val_loader)) if self.val_loader is not None else None

    def _targets_for_loss(self, targets):
        if self.is_usfm_official:
            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            return targets.long()
        return targets

    def _metric_tensors(self, outputs, targets):
        if self.is_usfm_official:
            if outputs.dim() == 4 and outputs.size(1) > 1:
                pred = outputs.argmax(dim=1, keepdim=True).float()
            else:
                pred = (outputs > 0.5).float()
            if targets.dim() == 3:
                target = targets.unsqueeze(1)
            else:
                target = targets
            return (pred > 0).float(), (target > 0).float()

        if self.output_is_prob:
            pred_probs = outputs
        else:
            pred_probs = torch.sigmoid(outputs)
        return pred_probs, targets

    def _update_and_check_best(self, val_log, current_epoch):
        """
        🚀 平均排名策略：综合考虑 Loss, Dice, HD95, IoU 决定最佳模型
        """
        # 1. 累积当前 epoch 的所有验证指标
        current_metrics = {'epoch': current_epoch, **val_log}
        self.metric_history.append(current_metrics)

        # 2. 转换为 DataFrame 方便批量计算排名
        history_df = pd.DataFrame(self.metric_history)

        rank_cols = []

        # Loss 和 HD95：数值越小越好 (ascending=True)
        if 'val_loss' in history_df.columns:
            history_df['loss_rank'] = history_df['val_loss'].rank(ascending=True, method='dense')
            rank_cols.append('loss_rank')
        if 'val_hd95_batch' in history_df.columns:
            history_df['hd95_rank'] = history_df['val_hd95_batch'].rank(ascending=True, method='dense')
            rank_cols.append('hd95_rank')

        # IoU 和 Dice：数值越大越好 (ascending=False)
        if 'val_iou_score' in history_df.columns:
            history_df['iou_rank'] = history_df['val_iou_score'].rank(ascending=False, method='dense')
            rank_cols.append('iou_rank')
        if 'val_dice_score' in history_df.columns:
            history_df['dice_rank'] = history_df['val_dice_score'].rank(ascending=False, method='dense')
            rank_cols.append('dice_rank')

        # 如果没有任何排名列被生成（防呆机制），默认当前就是最佳
        if not rank_cols:
            self.best_epoch_info = current_metrics
            return True

        # 3. 计算所有可用指标排名的平均值
        history_df['avg_rank'] = history_df[rank_cols].mean(axis=1)

        # 4. 找到平均排名数值最小（排名最靠前）的那一行
        best_epoch_idx = history_df['avg_rank'].idxmin()
        self.best_epoch_info = history_df.loc[best_epoch_idx].to_dict()

        # 5. 判断“历史最佳”是否就是“当前 Epoch”
        if self.best_epoch_info['epoch'] == current_epoch:
            return True

        return False

    def train(self):
        """Full training logic."""
        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            train_log = self._train_epoch(epoch)

            # 🌟 修复底层验证集空判问题
            valid_log = self._valid_epoch(epoch) if self.val_loader is not None else {}

            log = {'epoch': epoch, **train_log, **valid_log}

            # Print combined log
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            for key, value in log.items():
                if key != 'epoch':
                    print(f"    {key:15s}: {value:.4f}")

            # Wandb logging
            if wandb.run:
                wandb.log(log)
                self._log_validation_images(epoch)

            # Checkpoint and Early Stopping Logic
            if valid_log:
                # 🚀 使用综合排名机制判断是否是最佳模型
                is_new_best = self._update_and_check_best(valid_log, epoch)

                if is_new_best:
                    self.epochs_without_improvement = 0
                    self._save_checkpoint()
                    print(
                        f"🌟 Saved new best model at epoch {epoch}! (Avg Rank: {self.best_epoch_info.get('avg_rank', 0):.4f})")
                else:
                    self.epochs_without_improvement += 1
                    print(
                        f"⚠️ Early stopping counter: {self.epochs_without_improvement}/{self.early_stopping_patience}")

                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered after {epoch} epochs.")
                    print("🏆 Best Epoch Info:",
                          {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in self.best_epoch_info.items()})
                    break

        if wandb.run:
            wandb.run.summary.update(self.best_epoch_info)

    def _train_epoch(self, epoch):
        """Training logic for an epoch."""
        self.model.train()
        self.train_metrics.reset()

        desc = f"[Train Epoch {epoch}]"
        pbar = tqdm(self.train_loader, desc=desc, leave=False)

        for batch_idx, data in enumerate(pbar):
            inputs = data['image'].to(self.device)
            targets = data['mask'].to(self.device)
            loss_targets = self._targets_for_loss(targets)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # 🚀 修改点 3：Loss 计算与输出提取三分支
            if isinstance(outputs, dict) and "pred" in outputs:
                loss = self.criterion(outputs, loss_targets)
                outputs = outputs["pred"] # 提取出概率图用于后续评估

            elif isinstance(outputs, tuple) and len(outputs) == 2:
                out_main, out_aux = outputs
                loss_main = self.criterion(out_main, loss_targets)
                loss_aux = self.criterion(out_aux, loss_targets)
                aux_weight = self.config.get('usfm_aux_weight', 0.4)
                loss = loss_main + aux_weight * loss_aux
                outputs = out_main

            else:
                loss = self.criterion(outputs, loss_targets)

            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                if self.config.get('is_timm_scheduler', False):
                    if self.config.get('scheduler_update_per_iter', False):
                        self.lr_scheduler.step_update((epoch - 1) * len(self.train_loader) + batch_idx)
                    else:
                        self.lr_scheduler.step(epoch + batch_idx / len(self.train_loader))

            self.train_metrics.update('loss', loss.item(), n=targets.size(0))

            with torch.no_grad():
                outputs_detached = outputs.detach()

                # 🚀 修改点 4：配置驱动的概率校准，彻底解决数值盲猜的 Bug
                pred_probs, metric_targets = self._metric_tensors(outputs_detached, targets)

                # 计算 IoU
                batch_iou = iou_score(pred_probs, metric_targets)

                if batch_iou is not None:
                    if isinstance(batch_iou, tuple) and len(batch_iou) == 2:
                        iou_sum, iou_count = batch_iou
                        batch_iou = iou_sum / iou_count if iou_count > 0 else 0.0
                    elif isinstance(batch_iou, torch.Tensor):
                        batch_iou = batch_iou.float().mean().item()
                    elif isinstance(batch_iou, (np.ndarray, list)):
                        batch_iou = float(np.mean(batch_iou))
                    else:
                        batch_iou = float(batch_iou)

                    self.train_metrics.update('iou_score', batch_iou, n=targets.size(0))

            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in self.train_metrics.result().items()})

        if self.lr_scheduler is not None and not self.config.get('is_timm_scheduler', False):
            self.lr_scheduler.step()

        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        """Validate after training an epoch."""
        self.model.eval()
        self.valid_metrics.reset()

        desc = f"[Valid Epoch {epoch}]"
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc=desc, leave=False)):
                inputs = data['image'].to(self.device)
                targets = data['mask'].to(self.device)
                loss_targets = self._targets_for_loss(targets)

                outputs = self.model(inputs)

                # 🚀 修改点 5：验证集 Loss 计算与输出提取三分支
                if isinstance(outputs, dict) and "pred" in outputs:
                    loss = self.criterion(outputs, loss_targets)
                    outputs = outputs["pred"]
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    out_main, out_aux = outputs
                    loss = self.criterion(out_main, loss_targets)
                    outputs = out_main
                else:
                    loss = self.criterion(outputs, loss_targets)

                self.valid_metrics.update('loss', loss.item(), n=targets.size(0))

                # 🚀 修改点 6：验证集配置驱动的概率校准
                pred_probs, metric_targets = self._metric_tensors(outputs, targets)

                for met in self.metric_fns:
                    metric_value = met(pred_probs, metric_targets)

                    if metric_value is not None:
                        if isinstance(metric_value, tuple) and len(metric_value) == 2:
                            v_sum, v_count = metric_value
                            metric_value = v_sum / v_count if v_count > 0 else 0.0
                        elif isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.float().mean().item()
                        elif isinstance(metric_value, (np.ndarray, list)):
                            metric_value = float(np.mean(metric_value))
                        else:
                            metric_value = float(metric_value)

                        if not np.isnan(metric_value):
                            self.valid_metrics.update(met.__name__, metric_value, n=targets.size(0))

        return {f'val_{k}': v for k, v in self.valid_metrics.result().items()}

    def resume_checkpoint(self, resume_path):
        """Resumes from a saved checkpoint."""
        print(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['best_epoch_info'].get('epoch', 0) + 1
        self.best_epoch_info = checkpoint.get('best_epoch_info', {})
        self.model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}.")

    def _save_checkpoint(self):
        """Saves best model locally and logs it as a wandb artifact."""
        state = {
            'arch': type(self.model).__name__,
            'state_dict': self.model.state_dict(),
            'best_epoch_info': self.best_epoch_info,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / self.checkpoint_name)
        torch.save(state, filename)

        if wandb.run:
            artifact = wandb.Artifact(name=f"{wandb.run.name}-best-model", type="model")
            artifact.add_file(filename)

    def _log_validation_images(self, epoch):
        """Logs a fixed batch of validation images, masks, and predictions to wandb."""
        if not wandb.run: return
        if self.fixed_val_batch is None: return

        self.model.eval()
        images = self.fixed_val_batch['image'].to(self.device)
        gt_masks = self.fixed_val_batch['mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            if isinstance(outputs, dict) and "pred" in outputs:
                pred_logits = outputs["pred"]
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                pred_logits = outputs[0]
            else:
                pred_logits = outputs

            # 这里的可视化为了好看，统统过一遍 Sigmoid 也没关系，
            # 即使发生 double sigmoid 也就是亮一点
            if self.is_usfm_official:
                pred_masks, gt_masks = self._metric_tensors(pred_logits, gt_masks)
            else:
                pred_masks = torch.sigmoid(pred_logits)

        images_grid = make_grid(images, normalize=True)
        gt_masks_grid = make_grid(gt_masks, normalize=True)
        pred_masks_grid = make_grid(pred_masks, normalize=True)

        wandb.log({
            "validation_samples": [
                wandb.Image(images_grid.cpu().numpy().transpose(1, 2, 0), caption="Input Images"),
                wandb.Image(gt_masks_grid.cpu().numpy().transpose(1, 2, 0), caption="Ground Truth"),
                wandb.Image(pred_masks_grid.cpu().numpy().transpose(1, 2, 0), caption=f"Predictions (Epoch {epoch})")
            ]
        }, step=epoch)
