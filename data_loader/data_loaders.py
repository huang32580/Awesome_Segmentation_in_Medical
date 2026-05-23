# data_loader/data_loaders.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations import Compose
# Assume these are in data.preprocessing, as in the original code
from data.preprocessing.preprocess import get_preprocessing_transform
from data.preprocessing.augmentation import get_augmentation_transform


class BUSImageDataset(Dataset):
    def __init__(self, df, transform=None, label_map=None, pipeline='awesome'):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or {'benign': 0, 'malignant': 1, 'normal': 2}
        self.pipeline = pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.pipeline == 'usfm_official':
            image = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(row['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(row['image_path'])
        mask = None
        if pd.notnull(row.get('mask_path', None)):
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

            # --- FIX: Ensure mask and image have the same dimensions ---
            # This handles inconsistencies in the source dataset files.
            image_hw = image.shape[:2]
            if mask is not None and mask.shape != image_hw:
                mask = cv2.resize(mask, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            # For masks that don't exist, pass a placeholder to albumentations
            placeholder_mask = mask if mask is not None else image[..., 0] if image.ndim == 3 else image
            augmented = self.transform(image=image, mask=placeholder_mask)
            image = augmented['image']
            if mask is not None:
                mask = augmented['mask']

        if self.pipeline == 'usfm_official':
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if mask is not None:
                mask = torch.from_numpy((mask > 0).astype('int64'))
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
                mask = (mask > 0.5).float()

        label = self.label_map.get(str(row['label']).lower(), -1)

        return {
            "image": image,
            "mask": mask if mask is not None else torch.empty(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BUSDataLoader(DataLoader):
    """PyTorch DataLoader Factory for Breast US images."""

    def __init__(
            self, df, batch_size, split='1', is_test=False, num_workers=0,
            # === 增加 use_pad 参数，默认保持原来的 True 行为 ===
            label_map=None, target_size=512, padding_color=0, augment=True, use_pad=True,
            pipeline='awesome'
    ):
        # Handle test mode automatically
        if is_test:
            shuffle = False
            augment = False
        else:
            shuffle = True

        # === 将 use_pad 参数传递给 get_preprocessing_transform ===
        if pipeline == 'usfm_official':
            transforms_list = [A.Resize(width=target_size, height=target_size)]
            if augment:
                transforms_list.extend([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ])
            transforms_list.extend([
                A.ToFloat(max_value=255),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1,
                ),
            ])
            transform = A.Compose(transforms_list, additional_targets={'mask': 'mask'})
        else:
            transforms_list = [get_preprocessing_transform(target_size, padding_color, use_pad=use_pad)]

            if augment:
                transforms_list.append(get_augmentation_transform())
            transform = Compose(transforms_list, additional_targets={'mask': 'mask'})
        # Filter dataframe for the specified split
        if not is_test:
            # In k-fold, the train split contains all data EXCEPT the current validation fold and the test set
            df_split = df[(df['split'] != str(split)) & (df['split'] != 'test')].copy()
        else:
            # For validation or test loader, we just need the specified split
            df_split = df[df['split'] == str(split)].copy()

        dataset = BUSImageDataset(df_split, transform=transform, label_map=label_map, pipeline=pipeline)

        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True
        )


