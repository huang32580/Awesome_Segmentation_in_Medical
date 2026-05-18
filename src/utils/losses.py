# src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

try:
    from .LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['DiceBCELoss', 'LovaszHingeLoss']

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
    

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE
        return F_loss.mean()

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        # loss를 device로 이동
        self.loss.pos_weight = self.loss.pos_weight.to(inputs.device)
        return self.loss(inputs, targets)

class EdgeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sobel = kornia.filters.Sobel()

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        pred_edge = self.sobel(inputs_sigmoid)
        target_edge = self.sobel(targets)
        return F.l1_loss(pred_edge, target_edge)

def compute_sdf(mask):

    mask = mask.cpu().numpy()
    sdf = np.zeros_like(mask, dtype=np.float32)
    for b in range(mask.shape[0]):
        posmask = mask[b,0].astype(bool)
        if posmask.any(): # 마스크가 비어있지 않을 때만 계산
            negmask = ~posmask
            sdf[b,0] = edt(negmask) - edt(posmask)
    return torch.from_numpy(sdf)

class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, gt):

        pred_sigmoid = torch.sigmoid(pred_logits) 
        
        sdf_gt = compute_sdf(gt).to(gt.device)
        loss = torch.mean((pred_sigmoid - gt) * sdf_gt)
        return loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, bce_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets, smooth=1e-7):
        
        self.bce.pos_weight = self.bce.pos_weight.to(inputs.device)

        # BCE loss
        bce_loss = self.bce(inputs, targets)

        # Dice loss
        inputs_sig = torch.sigmoid(inputs)
        # Flatten a N-D Tensor to 1-D Tensor
        intersection = (inputs_sig.flatten() * targets.flatten()).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.flatten().sum() + targets.flatten().sum() + smooth)
        
        return self.bce_weight * bce_loss + self.weight * dice_loss


class ComboLossHD(nn.Module):

    def __init__(self, alpha=0.8, gamma=2, edge_weight=1.0, boundary_weight=1.0, ce_weight=1.0, pos_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.edge = EdgeLoss()
        self.boundary = BoundaryLoss()
        self.ce = WeightedCrossEntropyLoss(pos_weight=pos_weight)
        self.edge_weight = edge_weight
        self.boundary_weight = boundary_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        edge_loss = self.edge(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        ce_loss = self.ce(inputs, targets)

        total_loss = (
            focal_loss
            + self.edge_weight * edge_loss
            + self.boundary_weight * boundary_loss
            + self.ce_weight * ce_loss
        )
        return total_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional
from torch import Tensor
import torchvision


# =========================================================================
# 以下为移植自 USFM 官方的 ATMLoss (去除了 mmseg 注册器依赖)
# =========================================================================

def is_dist_avail_and_initialized():
    if not dist.is_available(): return False
    if not dist.is_initialized(): return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized(): return 1
    return dist.get_world_size()


def dice_loss_atm(inputs, targets, num_masks):
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        tensor = torch.zeros(batch_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=tensor_list[0].device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, weight_dict, losses, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = outputs["pred_logits"]
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        # 🚀 修复设备问题：获取当前数据所在的 GPU
        device = src_logits.device

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        target_classes[batch_idx, src_idx] = target_classes_o

        # 🚀 确保 empty_weight 也在同一个 GPU 上
        empty_weight = self.empty_weight.to(device)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear",
                                  align_corners=False)[:, 0].flatten(1)
        target_masks = target_masks.flatten(1).view(src_masks.shape)

        return {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss_atm(src_masks, target_masks, num_masks),
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        labels = [x["labels"] for x in targets]
        indices = [[label, torch.arange(len(label))] for label in labels]
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
            losses.update(loss_map[loss](outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = loss_map[loss](aux_outputs, targets, indices, num_masks)
                    losses.update({k + f"_{i}": v for k, v in l_dict.items()})
        return losses


class ATMLoss(nn.Module):
    def __init__(self, num_classes=1, dec_layers=3, mask_weight=20.0, dice_weight=1.0, cls_weight=1.0, loss_weight=1.0,
                 ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        weight_dict = {"loss_ce": cls_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes, weight_dict=weight_dict, losses=["labels", "masks"])
        self.loss_weight = loss_weight

    def prepare_targets(self, targets):
        new_targets = []
        # 【新增】🚀 强制二值化收束：无论原始前景像素是 1 还是 255，统统归一化为 0(背景) 和 1(病灶)
        targets = (targets > 0).long()
        for targets_per_image in targets:
            # 确保 mask 为整数以便提取
            t_img = targets_per_image.long()
            gt_cls = t_img.unique()

            # 🚀 修复越界核心：背景 (0) 不是目标！只提取 > 0 的病灶区域
            valid_cls = gt_cls[(gt_cls > 0) & (gt_cls != self.ignore_index)]

            masks = []
            labels = []
            for cls in valid_cls:
                masks.append(t_img == cls)
                # Query 索引从 0 开始，所以真实标签 1 需要映射到索引 0
                labels.append((cls - 1).long())

            if len(labels) == 0:
                # 如果这张图全是背景，没有病灶
                labels = torch.empty(0, dtype=torch.int64, device=targets.device)
                masks = torch.empty((0, targets.shape[-2], targets.shape[-1]), dtype=torch.bool, device=targets.device)
            else:
                labels = torch.stack(labels).to(targets.device)
                masks = torch.stack(masks, dim=0).to(targets.device)

            new_targets.append({"labels": labels, "masks": masks})
        return new_targets

    def forward(self, outputs, label):
        # 兼容 [B, 1, H, W] 格式的 label
        if label.dim() == 4 and label.size(1) == 1:
            label = label.squeeze(1)

        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                losses.pop(k)

        # 为了兼容框架，返回一个标量总 Loss 和字典
        total_loss = sum(losses.values())
        return total_loss