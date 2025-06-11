import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import os

class SkeletonFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

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
        features['shoulder_hip_ratio'] = shoulder_width / hip_width
        
        # ウエスト位置
        waist_height = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y + 
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        features['waist_position'] = waist_height
        
        # 胴体の縦横比
        torso_height = self._calculate_distance(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        )
        features['torso_ratio'] = torso_height / shoulder_width
        
        # 関節の突出度
        features['shoulder_prominence'] = self._calculate_joint_prominence(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        )
        
        features['hip_prominence'] = self._calculate_joint_prominence(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        )
        
        return features

    def _calculate_distance(self, point1, point2):
        """2点間の距離を計算"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def _calculate_joint_prominence(self, point1, point2):
        """関節の突出度を計算"""
        return abs(point1.z - point2.z)

class SkeletonClassifier:
    def __init__(self):
        self.feature_extractor = SkeletonFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }
        self.best_model = None

    def prepare_dataset(self, image_dir, labels):
        """データセットの準備"""
        print(f"データセット準備を開始します。入力ディレクトリ: {image_dir}")
        features_list = []
        image_paths = []
        
        # 各骨格タイプのディレクトリを処理
        for skeleton_type in ["straight", "wave", "natural"]:
            type_dir = os.path.join(image_dir, skeleton_type)
            if not os.path.exists(type_dir):
                print(f"ディレクトリが存在しません: {type_dir}")
                continue
                
            print(f"{skeleton_type} タイプの画像を処理します...")
            
            image_files = [f for f in os.listdir(type_dir) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"検出された画像ファイル数 ({skeleton_type}): {len(image_files)}")
            if image_files:
                print(f"最初の画像ファイル ({skeleton_type}): {image_files[0]}")

            for image_file in image_files:
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        image_path = os.path.join(type_dir, image_file)
                        print(f"画像を処理中: {image_path}")
                        features = self.feature_extractor.extract_features(image_path)
                        print(f"特徴量抽出完了: {image_path}")
                        features_list.append(features)
                        image_paths.append(image_path)
                    except Exception as e:
                        print(f"Error processing {image_file}: {e}")
                        continue
        
        X = pd.DataFrame(features_list)
        print(f"DataFrame作成完了。形状: {X.shape}")
        y = np.array([os.path.basename(os.path.dirname(path)) for path in image_paths])
        print(f"ラベル配列作成完了。形状: {y.shape}")
        
        # ラベルのエンコード
        print("ラベルをエンコードします。")
        y = self.label_encoder.fit_transform(y)
        print("ラベルエンコード完了。")
        
        # 特徴量の正規化
        print("特徴量を正規化します。")
        X_scaled = self.scaler.fit_transform(X)
        print(f"特徴量正規化完了。形状: {X_scaled.shape}")
        return X_scaled, y

    def train_and_evaluate(self, X, y):
        """モデルの学習と評価"""
        print("モデルの学習と評価を開始します。")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        for name, model in self.models.items():
            # クロスバリデーション
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"{name} CV scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # モデルの学習
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'model': model
            }
        
        # 最良のモデルを選択
        best_model_name = max(results, key=lambda x: results[x]['test_score'])
        self.best_model = results[best_model_name]['model']
        
        return results

    def save_model(self, path):
        """モデルの保存"""
        print(f"モデルを保存します: {path}")
        if self.best_model is None:
            raise ValueError("モデルが学習されていません")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, path)

    def load_model(self, path):
        """モデルの読み込み"""
        saved_data = joblib.load(path)
        self.best_model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.label_encoder = saved_data['label_encoder']

    def predict(self, image_path):
        """新しい画像の骨格タイプを予測"""
        features = self.feature_extractor.extract_features(image_path)
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        y_pred = self.best_model.predict(X_scaled)[0]
        return self.label_encoder.inverse_transform([y_pred])[0]

def main():
    # 使用例
    print("skeleton_classifier.py を実行します。")
    classifier = SkeletonClassifier()
    
    # データセットの準備
    image_dir = "data/augmented"  # 拡張画像のディレクトリ
    os.makedirs("models", exist_ok=True)  # モデル保存用ディレクトリの作成
    
    # データセットの準備と学習
    X, y = classifier.prepare_dataset(image_dir, None)  # labelsは不要になりました
    results = classifier.train_and_evaluate(X, y)
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    train_scores = [results[name]['train_score'] for name in model_names]
    test_scores = [results[name]['test_score'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Train Score')
    plt.bar(x + width/2, test_scores, width, label='Test Score')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    
    # グラフの保存
    plt.savefig('models/model_performance.png')
    plt.close()
    
    # モデルの保存
    classifier.save_model('models/skeleton_classifier.joblib')

if __name__ == "__main__":
    main()