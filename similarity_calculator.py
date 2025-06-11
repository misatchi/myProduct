import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
import cv2
from typing import List, Dict, Tuple
from feature_extractor import DetailedFeatureExtractor
import os

class SimilarityCalculator:
    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)
        self.feature_weights = {
            'shoulder_hip_ratio': 0.3,
            'waist_position': 0.2,
            'torso_ratio': 0.3,
            'shoulder_prominence': 0.1,
            'hip_prominence': 0.1
        }

    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """2つの特徴量ベクトル間の類似度を計算"""
        # 特徴量を配列に変換
        vec1 = self._features_to_vector(features1)
        vec2 = self._features_to_vector(features2)
        
        # コサイン類似度を計算
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        
        # 重み付け
        weighted_similarity = self._apply_weights(similarity, features1, features2)
        
        return weighted_similarity

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """特徴量辞書をベクトルに変換"""
        return np.array([features[key] for key in self.feature_weights.keys()])

    def _apply_weights(self, base_similarity: float, features1: Dict, features2: Dict) -> float:
        """特徴量の重みを適用"""
        weighted_diff = 0
        for feature, weight in self.feature_weights.items():
            diff = abs(features1[feature] - features2[feature])
            weighted_diff += diff * weight
        
        return base_similarity * (1 - weighted_diff)

    def find_similar_images(self, target_features: Dict, skeleton_type: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """類似画像を検索"""
        type_dir = self.features_dir / skeleton_type
        if not type_dir.exists():
            raise ValueError(f"骨格タイプ {skeleton_type} のディレクトリが存在しません")
        
        similarities = []
        for feature_file in type_dir.glob("*.json"):
            with open(feature_file, 'r') as f:
                features = json.load(f)
            
            similarity = self.calculate_similarity(target_features, features)
            image_path = str(feature_file).replace('.json', '.jpg')
            similarities.append((image_path, similarity))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def ensure_diversity(self, similar_images: List[Tuple[str, float]], min_similarity_diff: float = 0.1) -> List[Tuple[str, float]]:
        """推薦結果の多様性を確保"""
        diverse_results = []
        for image_path, similarity in similar_images:
            # 既存の結果との類似度をチェック
            if not diverse_results or all(
                abs(similarity - existing_sim) >= min_similarity_diff
                for _, existing_sim in diverse_results
            ):
                diverse_results.append((image_path, similarity))
            
            if len(diverse_results) >= 3:
                break
        
        return diverse_results

class RecommendationSystem:
    def __init__(self, features_dir: str, images_dir: str):
        self.similarity_calculator = SimilarityCalculator(features_dir)
        self.images_dir = Path(images_dir)
        self.feedback_weights = {}  # ユーザーフィードバックに基づく重み

    def get_recommendations(self, target_features: Dict, skeleton_type: str, confidence: float) -> List[Dict]:
        """骨格タイプに基づいて推薦を行う"""
        # 類似画像の検索
        similar_images = self.similarity_calculator.find_similar_images(
            target_features, skeleton_type
        )
        
        # 多様性を確保
        diverse_recommendations = self.similarity_calculator.ensure_diversity(similar_images)
        
        # 推薦結果の作成
        recommendations = []
        for image_path, similarity in diverse_recommendations:
            recommendations.append({
                'image_path': image_path,
                'similarity_score': similarity,
                'confidence': confidence
            })
        
        return recommendations

    def update_feedback_weights(self, image_path: str, is_helpful: bool):
        """ユーザーフィードバックに基づいて重みを更新"""
        if image_path not in self.feedback_weights:
            self.feedback_weights[image_path] = {'helpful': 0, 'total': 0}
        
        self.feedback_weights[image_path]['total'] += 1
        if is_helpful:
            self.feedback_weights[image_path]['helpful'] += 1

    def get_feedback_score(self, image_path: str) -> float:
        """フィードバックスコアを取得"""
        if image_path not in self.feedback_weights:
            return 0.5  # デフォルト値
        
        feedback = self.feedback_weights[image_path]
        if feedback['total'] == 0:
            return 0.5
        
        return feedback['helpful'] / feedback['total']

def main():
    augmentor = DataAugmentor()
    
    # 各骨格タイプのディレクトリを処理
    base_dir = "data/original"  # パスを修正
    for skeleton_type in ["straight", "wave", "natural"]:
        input_dir = os.path.join(base_dir, skeleton_type)
        if os.path.exists(input_dir):
            augmentor.process_directory(input_dir, skeleton_type)
        else:
            print(f"ディレクトリが存在しません: {input_dir}") 