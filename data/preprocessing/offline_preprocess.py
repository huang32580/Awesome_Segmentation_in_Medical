import cv2
import numpy as np
import os
from pathlib import Path


def process_and_crop(src_img_path, dst_img_path, dst_mask_path):
    # 1. 读取原图 (带红线)
    img = cv2.imread(str(src_img_path))
    if img is None:
        print(f"无法读取: {src_img_path}")
        return

    # 2. 提取红线并生成二值 Mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_lines = mask1 + mask2
    red_lines = cv2.morphologyEx(red_lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(red_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(red_lines)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, thickness=-1)

    # 3. 图像修复 (抹除红线避免模型作弊) 并转为灰度图
    clean_img = cv2.inpaint(img, red_lines, 3, cv2.INPAINT_TELEA)
    gray_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)

    # 4. Otsu 算法确定最大有效区域裁剪框
    gray_img_eq = cv2.equalizeHist(gray_img)
    _, thresh = cv2.threshold(gray_img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    crop_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if crop_contours:
        c_crop = max(crop_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c_crop)
        # 核心：使用同一组 (x, y, w, h) 同步裁剪灰度原图和 Mask
        cropped_img = gray_img[y:y + h, x:x + w]
        cropped_mask = final_mask[y:y + h, x:x + w]
    else:
        cropped_img = gray_img
        cropped_mask = final_mask

    # 5. 保存结果
    cv2.imwrite(str(dst_img_path), cropped_img)
    cv2.imwrite(str(dst_mask_path), cropped_mask)


def main():
    # 你的原始文件夹路径
    raw_base_dir = Path("/hy-tmp/MyData")
    # 处理后的输出路径，建议放到你的项目 data/datasets/ 目录下
    out_base_dir = Path("/hy-tmp/datasets/MyData")

    phases = ["pre_chemo", "mid_chemo"]

    for phase in phases:
        raw_phase_dir = raw_base_dir / phase
        out_img_dir = out_base_dir / "images" / phase
        out_mask_dir = out_base_dir / "masks" / phase

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        for img_path in raw_phase_dir.glob("*.jpg"):
            dst_img = out_img_dir / f"{img_path.stem}.png"
            dst_mask = out_mask_dir / f"{img_path.stem}.png"
            process_and_crop(img_path, dst_img, dst_mask)
            print(f"Processed {img_path.name}")


if __name__ == "__main__":
    main()