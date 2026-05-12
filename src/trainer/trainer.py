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

        # 🔙 恢复为原版的保存策略变量
        self.mnt_best = np.inf  # 记录最低的 val_loss
        self.best_epoch_info = {}

        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.epochs_without_improvement = 0

        # 🌟 修复底层验证集空判问题
        self.fixed_val_batch = next(iter(self.val_loader)) if self.val_loader is not None else None

    def _update_and_check_best(self, val_log, current_epoch):
        """
        🔙 原版最佳模型评估策略：以 val_loss 为唯一标准（Loss 越低越好）
        """
        # 获取当前 epoch 的验证集 loss
        val_loss = val_log.get('val_loss', np.inf)

        # 判断是否比历史最好记录更低
        is_best = val_loss < self.mnt_best

        if is_best:
            # 更新记录
            self.mnt_best = val_loss
            self.best_epoch_info = {'epoch': current_epoch, **val_log}
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

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if isinstance(outputs, dict) and "pred" in outputs:
                loss = self.criterion(outputs, targets)
                outputs = outputs["pred"]

            elif isinstance(outputs, tuple) and len(outputs) == 2:
                out_main, out_aux = outputs
                loss_main = self.criterion(out_main, targets)
                loss_aux = self.criterion(out_aux, targets)
                aux_weight = self.config.get('usfm_aux_weight', 0.4)
                loss = loss_main + aux_weight * loss_aux
                outputs = out_main

            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                if self.config.get('is_timm_scheduler', False):
                    self.lr_scheduler.step(epoch + batch_idx / len(self.train_loader))

            self.train_metrics.update('loss', loss.item(), n=targets.size(0))

            with torch.no_grad():
                outputs_detached = outputs.detach()

                # 🚀 动态判断：防止 Double Sigmoid，同时拯救 CNN
                if outputs_detached.max() > 1.0 or outputs_detached.min() < 0.0:
                    pred_probs = torch.sigmoid(outputs_detached)
                else:
                    pred_probs = outputs_detached

                # 🚀 用经过概率校准的值计算 IoU
                batch_iou = iou_score(pred_probs, targets)

                if batch_iou is not None:
                    # ============ 🚀 健壮的指标解析逻辑 ============
                    if isinstance(batch_iou, tuple) and len(batch_iou) == 2:
                        iou_sum, iou_count = batch_iou
                        batch_iou = iou_sum / iou_count if iou_count > 0 else 0.0
                    elif isinstance(batch_iou, torch.Tensor):
                        batch_iou = batch_iou.float().mean().item()
                    elif isinstance(batch_iou, (np.ndarray, list)):
                        batch_iou = float(np.mean(batch_iou))
                    else:
                        batch_iou = float(batch_iou)
                    # ===============================================

                    # 🚀 补全权重参数 n
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

                outputs = self.model(inputs)

                if isinstance(outputs, dict) and "pred" in outputs:
                    loss = self.criterion(outputs, targets)
                    outputs = outputs["pred"]
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    out_main, out_aux = outputs
                    loss = self.criterion(out_main, targets)
                    outputs = out_main
                else:
                    loss = self.criterion(outputs, targets)

                # 🚀 补全验证 loss 的权重参数 n
                self.valid_metrics.update('loss', loss.item(), n=targets.size(0))

                # 🚀 动态判断：防止 Double Sigmoid，同时拯救 CNN
                if outputs.max() > 1.0 or outputs.min() < 0.0:
                    pred_probs = torch.sigmoid(outputs)
                else:
                    pred_probs = outputs

                for met in self.metric_fns:
                    # 🚀 用经过概率校准的值计算指标
                    metric_value = met(pred_probs, targets)

                    if metric_value is not None:
                        # ====== 智能解析不同指标返回格式 ======
                        if isinstance(metric_value, tuple) and len(metric_value) == 2:
                            v_sum, v_count = metric_value
                            metric_value = v_sum / v_count if v_count > 0 else 0.0
                        elif isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.float().mean().item()
                        elif isinstance(metric_value, (np.ndarray, list)):
                            metric_value = float(np.mean(metric_value))
                        else:
                            metric_value = float(metric_value)
                        # ====================================

                        if not np.isnan(metric_value):
                            # 🚀 补全所有指标的权重参数 n
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