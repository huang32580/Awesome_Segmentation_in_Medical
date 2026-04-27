#data/preprocessing/preprocess.py
import albumentations as A
import cv2


def get_preprocessing_transform(target_size=512, padding_color=0, use_pad=True):
    """
    target_size: int, desired output size (e.g., 224, 256, 448, 512)
    padding_color: int or (int, int, int), value for padding
    use_pad: bool, 是否使用等比例缩放+Padding。如果为False，则直接暴力拉伸到 target_size
    """

    def get_transform():
        if use_pad:
            # 方式一：保持长宽比，不足的部分黑边 Padding（原版逻辑）
            return A.Compose([
                A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LANCZOS4),
                A.PadIfNeeded(
                    min_height=target_size, min_width=target_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=padding_color,  # 图像 padding 颜色
                    mask_value=0,  # 掩码 padding 黑色 (背景)
                    position='center'
                ),
            ], additional_targets={'mask': 'mask'})
        else:
            # 方式二：直接暴力拉伸到目标尺寸（适配你的 JEPA 预训练逻辑）
            return A.Compose([
                A.Resize(height=target_size, width=target_size, interpolation=cv2.INTER_LANCZOS4)
            ], additional_targets={'mask': 'mask'})

    return get_transform() 