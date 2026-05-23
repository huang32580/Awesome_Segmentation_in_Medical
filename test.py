# test.py
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from thop import profile
from pathlib import Path
from collections import defaultdict
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning)

from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.utils.parse_config import ConfigParser
from src.utils.metrics import pixel_accuracy, dice_score, hd95_batch, iou_score
import src.models.cnn_based as cnn_models
import src.models.ViT_based as transformer_models


def main(config):
    """Main function to run the evaluation pipeline."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare Data
    data_config = config['data']
    preparer = PrepareDataset()
    all_dfs = []
    for name in data_config['datasets']:
        csv_path = preparer.data_dir / f"{name}.csv"
        if csv_path.exists():
            all_dfs.append(pd.read_csv(csv_path))
        else:
            raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.concat(all_dfs, ignore_index=True)
    model_type = config['arch']['type']
    usfm_args = config.config.get('usfm_args', {})
    is_official_usfm = (model_type == 'USFM' and usfm_args.get('mode', 'local') == 'official')
    decoder_type = usfm_args.get('decoder_type', '')
    data_pipeline = 'usfm_official' if is_official_usfm else 'awesome'
    num_classes = 2 if is_official_usfm else 1

    loader_args = {
        'batch_size': 1,  # 测试时通常 bs 设为 1
        'num_workers': data_config.get('num_workers', 4),
        'target_size': data_config['target_size'],
        'use_pad': data_config.get('use_pad', True),
        'pipeline': data_pipeline,
    }

    # 提取折数以区分不同的测试集，假设模型名字里有 fold 标识
    resume_path = str(config.resume)
    fold_str = '1'
    if 'fold' in resume_path:
        fold_str = resume_path.split('fold')[1][0]

    # ================= 🚀 测试集划分问题 =================
    # 在标准 5 折交叉验证中，'test' 是完全独立且未参与划分的盲测集
    print(f"[*] Testing on Held-out Test Set (Model trained on Fold: {fold_str})")

    # 获取真正的 test 集，而不是当前 fold 的验证集
    eval_split = 'test'
    test_df = df[df['split'] == eval_split].copy()

    # 防御性编程：万一你的 CSV 没有叫 'test' 的划分，退回使用验证集（避免报错崩溃）
    if len(test_df) == 0:
        print("⚠️ 警告：CSV中未发现 split=='test' 的数据！回退使用当前折的验证集作为测试。")
        eval_split = fold_str
        test_df = df[df['split'] == eval_split].copy()

    test_loader = BUSDataLoader(df, **loader_args, split=eval_split, is_test=True)

    # Initialize Model
    if hasattr(cnn_models, model_type):
        model = config.init_obj('arch', cnn_models)
    elif model_type in ["TransUnet", "SwinUnet", "MedT", "JEPA_UPerNet", "USFM"]:
        model = transformer_models.get_transformer_based_model(
            model_name=model_type,
            config=config.config,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Model type '{model_type}' not found.")

    # ==============================================================
    # 🚀 健壮性改进：提前解析配置，决定输出是否已经是概率图
    # ==============================================================
    # 只有 SegViT(ATMHead) 出来的 pred 是 0-1 概率图，其他都是 Logits
    output_is_prob = (model_type == 'USFM' and decoder_type == 'SegViT' and not is_official_usfm)

    # Load Checkpoint
    print(f"Loading checkpoint: {resume_path} ...")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Trackers for Metrics
    total_metrics_sum = defaultdict(float)
    total_metrics_count = defaultdict(int)

    desc = "[Testing]"
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc=desc)):
            inputs = data['image'].to(device)
            targets = data['mask'].to(device)

            # ==============================================================
            # 1. 将输入送入模型，拿到初始输出
            # ==============================================================
            outputs = model(inputs)

            # ==============================================================
            # 2. 兼容各种模型的输出格式（字典、元组等）
            # ==============================================================
            if isinstance(outputs, dict) and "pred" in outputs:
                # 针对 SegViT(ATMHead) 字典输出
                outputs = outputs["pred"]
            elif isinstance(outputs, tuple) and len(outputs) >= 2:
                # 针对 UPerNet 等元组输出，测试时只用主输出
                outputs = outputs[0]

            # ==============================================================
            # 3. 脱离计算图
            # ==============================================================
            outputs = outputs.detach()

            # ==============================================================
            # 4. 🚀 配置驱动的概率校准，彻底解决数值盲猜的 Bug
            # ==============================================================
            if is_official_usfm:
                if outputs.dim() == 4 and outputs.size(1) > 1:
                    outputs = outputs.argmax(dim=1, keepdim=True).float()
                else:
                    outputs = (outputs > 0.5).float()
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                targets = (targets > 0).float()
            elif output_is_prob:
                # SegViT 的输出已经是 0-1 的概率图，直接使用
                outputs = outputs
            else:
                # UPerNet 或传统 CNN 的输出是 Logits，需要加 Sigmoid
                outputs = torch.sigmoid(outputs)

            # --- 定义一个万能提取器，强行把各种输出转成 (Sum, Count) ---
            def parse_metric(metric_out, b_size):
                if metric_out is None: return 0.0, 0
                if isinstance(metric_out, tuple) and len(metric_out) == 2:
                    return metric_out[0], metric_out[1]  # 本身就是 (sum, count)
                # 如果传回来的是均值(浮点数/Tensor)，就乘以 b_size 逆向还原为总和
                val = metric_out.float().mean().item() if isinstance(metric_out, torch.Tensor) else float(
                    np.mean(metric_out))
                return val * b_size, b_size

            curr_b_size = inputs.size(0)

            # Calculate and Unpack
            pa_sum, pa_count = parse_metric(pixel_accuracy(outputs, targets), targets.numel())
            dsc_sum, dsc_count = parse_metric(dice_score(outputs, targets), curr_b_size)
            hd95_sum, hd95_count = parse_metric(hd95_batch(outputs, targets), curr_b_size)
            iou_sum, iou_count = parse_metric(iou_score(outputs, targets), curr_b_size)

            # Accumulate metrics
            total_metrics_sum['PA'] += pa_sum
            total_metrics_count['PA'] += pa_count

            total_metrics_sum['DSC'] += dsc_sum
            total_metrics_count['DSC'] += dsc_count

            if hd95_count > 0 and not pd.isna(hd95_sum):
                total_metrics_sum['HD95'] += hd95_sum
                total_metrics_count['HD95'] += hd95_count

            total_metrics_sum['IoU'] += iou_sum
            total_metrics_count['IoU'] += iou_count

    # Calculate Final Averages
    final_results = {}
    for key in ['DSC', 'HD95', 'IoU', 'PA']:
        if total_metrics_count[key] > 0:
            final_results[key] = total_metrics_sum[key] / total_metrics_count[key]
        else:
            final_results[key] = 0.0

    # Calculate FLOPs safely. Count parameters manually because thop can under-count
    # custom Transformer/dict-output modules.
    flops = 0
    params = sum(p.numel() for p in model.parameters())
    try:
        dummy_channels = 3 if is_official_usfm else 1 if data_config.get('in_channels', 1) == 1 else 3
        dummy_input = torch.randn(1, dummy_channels,
                                  data_config['target_size'], data_config['target_size']).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"\n[Warning] thop profile failed to calculate FLOPs: {e}")

    # Print Final Results
    print("\n--- Test Results ---")
    for key, value in final_results.items():
        print(f"{key}: {value:.4f}")
    if flops > 0:
        print(f"GFLOPs: {flops / 1e9:.2f}")
    print(f"Params: {params / 1e6:.2f}M")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Breast Ultrasound Segmentation Testing')
    args.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint to test')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (optional, will read from ckpt if None)')

    # 允许通过 ConfigParser 解析
    config = ConfigParser.from_args(args)
    main(config)
