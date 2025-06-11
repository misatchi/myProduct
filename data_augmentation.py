import cv2
import numpy as np
import os
from tqdm import tqdm
import albumentations as A
from pathlib import Path

class DataAugmentor:
    def __init__(self, output_dir="data/augmented"):
        self.output_dir = output_dir
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomGamma(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        for skeleton_type in ["straight", "wave", "natural"]:
            os.makedirs(os.path.join(output_dir, skeleton_type), exist_ok=True)

    def augment_image(self, image_path, skeleton_type, num_augmentations=3):
        """画像を拡張して保存する"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"画像を読み込めません: {image_path}")
            return
        
        # 元の画像を保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(
            self.output_dir,
            skeleton_type,
            f"{base_name}_original.jpg"
        )
        cv2.imwrite(output_path, image)
        
        # 拡張画像の生成と保存
        for i in range(num_augmentations):
            augmented = self.transform(image=image)
            augmented_image = augmented["image"]
            
            output_path = os.path.join(
                self.output_dir,
                skeleton_type,
                f"{base_name}_aug_{i+1}.jpg"
            )
            cv2.imwrite(output_path, augmented_image)

    def process_directory(self, input_dir, skeleton_type, num_augmentations=3):
        """ディレクトリ内の全画像を処理"""
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc=f"Processing {skeleton_type} images"):
            image_path = os.path.join(input_dir, image_file)
            self.augment_image(image_path, skeleton_type, num_augmentations)

def main():
    # 使用例
    augmentor = DataAugmentor()
    
    # 各骨格タイプのディレクトリを処理
    base_dir = "data/original"
    for skeleton_type in ["straight", "wave", "natural"]:
        input_dir = os.path.join(base_dir, skeleton_type)
        if os.path.exists(input_dir):
            augmentor.process_directory(input_dir, skeleton_type)
        else:
            print(f"ディレクトリが存在しません: {input_dir}")

if __name__ == "__main__":
    main() 