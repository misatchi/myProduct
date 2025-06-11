# analyze_skeleton_data.py

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

class SkeletonAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image_path):
        """画像から骨格のランドマークを抽出する"""
        image = cv2.imread(image_path)
        if image is None:
            # print(f"警告: 画像を読み込めませんでした: {image_path}")
            return None, None
        
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            # print(f"警告: 骨格を検出できませんでした: {image_path}")
            return None, image

        # ランドマークの座標を抽出 (x, y, z)
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])

        return landmarks, image

    def normalize_landmarks(self, landmarks):
        """ランドマーク座標を正規化する"""
        if landmarks is None or len(landmarks) < 33:
            return None

        # 正規化の基準点を定義（例: 首の中点）
        # Mediapipe landmark indices: 11 (left_shoulder), 12 (right_shoulder)
        # 首のランドマークはないため、肩の中点を代用
        # base_point = (landmarks[11] + landmarks[12]) / 2.0

        # または、胴体の中心（肩の中点と腰の中点の中間）を基準点とする
        # 23 (left_hip), 24 (right_hip)
        shoulder_mid = (landmarks[11] + landmarks[12]) / 2.0
        hip_mid = (landmarks[23] + landmarks[24]) / 2.0
        base_point = (shoulder_mid + hip_mid) / 2.0
        
        # 全身のスケールを計算（例: 首の中点から腰の中点までのY方向距離）
        scale = np.linalg.norm(shoulder_mid - hip_mid)

        if scale < 1e-6: # スケールが小さすぎる場合は正規化しない
             # print("警告: スケールが小さすぎます")
             return None

        # 基準点からの相対座標に変換し、スケールで割る
        normalized_landmarks = (landmarks - base_point) / scale

        return normalized_landmarks

    def calculate_features(self, normalized_landmarks):
        """正規化されたランドマークから特徴量を計算する"""
        if normalized_landmarks is None or len(normalized_landmarks) < 33:
            return None

        # 例としていくつかの特徴量を計算
        # 肩幅 (11, 12)
        shoulder_width = np.linalg.norm(normalized_landmarks[11] - normalized_landmarks[12])
        # 腰幅 (23, 24)
        hip_width = np.linalg.norm(normalized_landmarks[23] - normalized_landmarks[24])
        # 胴体の縦方向の長さ (肩の中点と腰の中点の距離)
        shoulder_mid = (normalized_landmarks[11] + normalized_landmarks[12]) / 2.0
        hip_mid = (normalized_landmarks[23] + normalized_landmarks[24]) / 2.0
        torso_length = np.linalg.norm(shoulder_mid - hip_mid)
        # 胴体の縦横比 (torso_length / (shoulder_width + hip_width)/2.0 など)
        # 簡単のために torso_length / shoulder_width とする
        torso_aspect_ratio = torso_length / shoulder_width if shoulder_width > 1e-6 else 0
        
        # 肩幅と腰幅の比率
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 1e-6 else 0

        # 他にも様々な特徴量が考えられます（例：腕の長さ、脚の長さなど）
        # 例：左腕の長さ (11->13->15)
        left_arm_length = np.linalg.norm(normalized_landmarks[11] - normalized_landmarks[13]) + \
                          np.linalg.norm(normalized_landmarks[13] - normalized_landmarks[15])
        # 例：右腕の長さ (12->14->16)
        right_arm_length = np.linalg.norm(normalized_landmarks[12] - normalized_landmarks[14]) + \
                           np.linalg.norm(normalized_landmarks[14] - normalized_landmarks[16])
        # 例：左脚の長さ (23->25->27)
        left_leg_length = np.linalg.norm(normalized_landmarks[23] - normalized_landmarks[25]) + \
                          np.linalg.norm(normalized_landmarks[25] - normalized_landmarks[27])
        # 例：右脚の長さ (24->26->28)
        right_leg_length = np.linalg.norm(normalized_landmarks[24] - normalized_landmarks[26]) + \
                           np.linalg.norm(normalized_landmarks[26] - normalized_landmarks[28])

        features = {
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'torso_length': torso_length,
            'torso_aspect_ratio': torso_aspect_ratio,
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'left_arm_length': left_arm_length,
            'right_arm_length': right_arm_length,
            'left_leg_length': left_leg_length,
            'right_leg_length': right_leg_length,
        }

        return features

    def analyze_data(self, data_dir="data", output_csv="skeleton_features.csv"):
        """データフォルダを分析し、特徴量をCSVに保存する"""
        all_features = []
        
        # データディレクトリ内の各タイプフォルダを走査
        for skeleton_type in os.listdir(data_dir):
            type_dir = os.path.join(data_dir, skeleton_type)
            if os.path.isdir(type_dir):
                print(f"Analyzing {skeleton_type} type...")
                
                # タイプフォルダ内の画像を走査
                for image_file in os.listdir(type_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(type_dir, image_file)
                        
                        # ランドマーク抽出
                        landmarks, _ = self.extract_landmarks(image_path)
                        
                        if landmarks is not None:
                            # ランドマーク正規化
                            normalized_landmarks = self.normalize_landmarks(landmarks)
                            
                            if normalized_landmarks is not None:
                                # 特徴量計算
                                features = self.calculate_features(normalized_landmarks)
                                
                                if features is not None:
                                    features['type'] = skeleton_type
                                    features['image_file'] = os.path.join(skeleton_type, image_file)
                                    all_features.append(features)

        # 特徴量をPandas DataFrameに変換
        features_df = pd.DataFrame(all_features)

        # 特徴量の統計情報を表示
        if not features_df.empty:
            print("\n--- 特徴量の統計情報（平均と標準偏差）---")
            for skeleton_type in features_df['type'].unique():
                print(f"\nType: {skeleton_type}")
                print(features_df[features_df['type'] == skeleton_type].describe().loc[['mean', 'std']])

            # 特徴量データをCSVに保存
            features_df.to_csv(output_csv, index=False)
            print(f"\n特徴量データを {output_csv} に保存しました。")
        else:
            print("特徴量を抽出できる画像が見つかりませんでした。")

if __name__ == "__main__":
    # 使用例
    # 以下のディレクトリ構造で画像を配置してください：
    # ./data/
    #   straight/
    #     image1.jpg
    #     image2.png
    #     ...
    #   wave/
    #     imageA.jpeg
    #     imageB.bmp
    #     ...
    #   natural/
    #     ...

    analyzer = SkeletonAnalyzer()
    analyzer.analyze_data(data_dir="./data", output_csv="skeleton_features.csv") 