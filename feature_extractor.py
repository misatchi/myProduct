import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path

class DetailedFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.feature_weights = {
            'shoulder_hip_ratio': 0.3,
            'waist_position': 0.2,
            'torso_ratio': 0.3,
            'shoulder_prominence': 0.1,
            'hip_prominence': 0.1
        }

    def extract_features(self, image_path):
        """画像から骨格特徴量を抽出する"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            raise ValueError("骨格が検出できません")
        
        landmarks = results.pose_landmarks.landmark
        
        # 特徴量の計算
        features = {}
        
        # 肩幅とヒップ幅の比率
        shoulder_width = self._calculate_distance(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        )
        hip_width = self._calculate_distance(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        )
        features['shoulder_hip_ratio'] = float(shoulder_width / hip_width)
        
        # ウエスト位置
        waist_height = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y + 
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        features['waist_position'] = float(waist_height)
        
        # 胴体の縦横比
        torso_height = self._calculate_distance(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        )
        features['torso_ratio'] = float(torso_height / shoulder_width)
        
        # 関節の突出度
        features['shoulder_prominence'] = float(self._calculate_joint_prominence(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ))
        
        features['hip_prominence'] = float(self._calculate_joint_prominence(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        ))
        
        return features

    def _calculate_distance(self, point1, point2):
        """2点間の距離を計算"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def _calculate_joint_prominence(self, point1, point2):
        """関節の突出度を計算"""
        return abs(point1.z - point2.z)

    def process_directory(self, input_dir, output_dir):
        """ディレクトリ内の全画像を処理して特徴量を保存"""
        print(f"処理を開始します。入力ディレクトリ: {input_dir}, 出力ディレクトリ: {output_dir}")
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 各骨格タイプのディレクトリを処理
        for skeleton_type in ["straight", "wave", "natural"]:
            type_dir = os.path.join(input_dir, skeleton_type)
            if not os.path.exists(type_dir):
                print(f"ディレクトリが存在しません: {type_dir}")
                continue
            
            print(f"{skeleton_type} タイプの画像を処理します...")
            # 出力ディレクトリの作成
            output_type_dir = os.path.join(output_dir, skeleton_type)
            os.makedirs(output_type_dir, exist_ok=True)
            
            # 画像ファイルの処理
            image_files = [f for f in os.listdir(type_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"検出された画像ファイル数: {len(image_files)}")
            if image_files:
                print(f"最初の画像ファイル: {image_files[0]}")
            
            # for image_file in tqdm(image_files, desc=f"Processing {skeleton_type} images"):
            for image_file in image_files:
                try:
                    image_path = os.path.join(type_dir, image_file)
                    print(f"画像を処理中: {image_path}")
                    features = self.extract_features(image_path)
                    
                    # 特徴量をJSONファイルとして保存
                    output_file = os.path.join(
                        output_type_dir,
                        f"{os.path.splitext(image_file)[0]}.json"
                    )
                    
                    print(f"特徴量を保存中: {output_file}")
                    with open(output_file, 'w') as f:
                        json.dump(features, f, indent=2)
                        
                    print(f"特徴量を保存しました: {output_file}")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue

def main():
    # 使用例
    print("feature_extractor.py を実行します。")
    extractor = DetailedFeatureExtractor()
    
    # 入力ディレクトリ（拡張画像）と出力ディレクトリ（特徴量）の設定
    input_dir = "data/augmented"
    output_dir = "data/features"
    
    # 特徴量の抽出と保存
    extractor.process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()