# test.py
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from thop import profile
from pathlib import Path
from collections import defaultdict
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

    loader_args = {
        'batch_size': 1,  # 测试时通常 bs 设为 1
        'num_workers': data_config.get('num_workers', 4),
        'target_size': data_config['target_size'],
        'use_pad': data_config.get('use_pad', True)
    }

    # 提取折数以区分不同的测试集，假设模型名字里有 fold 标识
    resume_path = str(config.resume)
    fold_str = '1'
    if 'fold' in resume_path:
        fold_str = resume_path.split('fold')[1][0]

    print(f"[*] Testing on Fold: {fold_str}")
    test_loader = BUSDataLoader(df[df['split'] == fold_str].copy(), **loader_args, split=fold_str, is_test=True)

    # Initialize Model
    model_type = config['arch']['type']
    if hasattr(cnn_models, model_type):
        model = config.init_obj('arch', cnn_models)
    elif model_type in ["TransUnet", "SwinUnet", "MedT", "JEPA_UPerNet", "USFM_UPerNet", "USFM_SegmentationModel"]:
        model = transformer_models.get_transformer_based_model(
            model_name=model_type,
            config=config.config,
            num_classes=1
        )
    else:
        raise ValueError(f"Model type '{model_type}' not found.")

    # Load Checkpoint
    print(f"Loading checkpoint: {resume_path} ...")
    checkpoint = torch.load(resume_path, map_location=device)
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

            outputs = model(inputs)

            # ================= 🚀 核心修改：兼容各种模型的输出 =================
            if isinstance(outputs, dict) and "pred" in outputs:
                # 针对 SegViT(ATMHead) 字典输出
                outputs = outputs["pred"]
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                # 针对 UPerNet (主输出, 辅助输出) 元组，测试时只用主输出
                outputs = outputs[0]
            # =================================================================

            # Detach for metric computation
            outputs = outputs.detach()

            # Calculate metrics
            batch_pa = pixel_accuracy(outputs, targets)
            batch_dsc = dice_score(outputs, targets)
            batch_hd95 = hd95_batch(outputs, targets)
            batch_iou = iou_score(outputs, targets)

            # Accumulate metrics
            pa_total_pixels = targets.numel()
            if batch_dsc is not None:
                total_metrics_sum['DSC'] += batch_dsc * inputs.size(0)
                total_metrics_count['DSC'] += inputs.size(0)
            if batch_hd95 is not None and not pd.isna(batch_hd95):
                total_metrics_sum['HD95'] += batch_hd95 * inputs.size(0)
                total_metrics_count['HD95'] += inputs.size(0)
            if batch_iou is not None:
                total_metrics_sum['IoU'] += batch_iou * inputs.size(0)
                total_metrics_count['IoU'] += inputs.size(0)
            if batch_pa is not None:
                total_metrics_sum['PA'] += batch_pa * pa_total_pixels
                total_metrics_count['PA'] += pa_total_pixels

    # Calculate Final Averages
    final_results = {}
    for key in ['DSC', 'HD95', 'IoU', 'PA']:
        if total_metrics_count[key] > 0:
            final_results[key] = total_metrics_sum[key] / total_metrics_count[key]
        else:
            final_results[key] = 0.0

    # Calculate FLOPs and Parameters safely
    flops, params = 0, 0
    try:
        dummy_input = torch.randn(1, 1 if data_config.get('in_channels', 1) == 1 else 3,
                                  data_config['target_size'], data_config['target_size']).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"\n[Warning] thop profile failed to calculate FLOPs: {e}")
        # params can still be calculated manually
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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