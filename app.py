from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from feature_extractor import DetailedFeatureExtractor
from similarity_calculator import SimilarityCalculator, RecommendationSystem
from skeleton_classifier import SkeletonClassifier
import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import traceback # エラーの詳細表示用

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_replace_me' # セッション用シークレットキーを設定してください

# アップロードされたファイルの保存先
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FEATURES_DIR = 'data/features'
AUGMENTED_IMAGES_DIR = 'data/augmented'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 必要なディレクトリの作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)
for skeleton_type in ["straight", "wave", "natural"]:
    os.makedirs(os.path.join(FEATURES_DIR, skeleton_type), exist_ok=True)
    os.makedirs(os.path.join(AUGMENTED_IMAGES_DIR, skeleton_type), exist_ok=True)

# アプリケーションの初期化
feature_extractor = DetailedFeatureExtractor()
classifier = SkeletonClassifier()
recommendation_system = RecommendationSystem(
    features_dir=FEATURES_DIR,
    images_dir=AUGMENTED_IMAGES_DIR
)

# モデルの読み込み
try:
    classifier.load_model("models/skeleton_classifier.joblib")
except FileNotFoundError:
    print("警告: モデルファイルが見つかりません。モデル学習を実行してください。")
    # モデルが存在しない場合のフラグを設定するなど、適切にハンドリング
    # 現在は分析時にエラーとする
    pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def analyze():
    if request.method == "GET":
        return render_template('upload.html')
    
    elif request.method == "POST":
        if 'file' not in request.files:
            return jsonify({'error': 'ファイルがアップロードされていません'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        if file and allowed_file(file.filename):
            # ファイルの保存
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # 特徴量の抽出
                features = feature_extractor.extract_features(filepath)
                
                # 骨格タイプの判別
                predicted_type = classifier.predict(filepath)
                
                # 判別の確信度計算 (簡易化のためここでは割愛)
                confidence = 1.0 # 仮の確信度
                
                # 診断結果に基づく推薦（既存機能）
                type_based_recommendations = recommendation_system.get_recommendations(
                    features, predicted_type, confidence
                )

                # 骨格タイプごとの説明
                reference_ex = {
                    'straight': [
                        '骨格ストレートは、全体的に立体感があり、筋肉も感じさせるメリハリがある体型！上重心なのが特徴！',
                    ],
                    'wave': [
                        '骨格ウェーブは、体は薄く、華奢で、柔らかな曲線を描いた体型！下重心なのが特徴！',
                    ],
                    'natural': [
                        '骨格ナチュラルは、筋肉や脂肪があまり感じらない、骨が太く大きく、関節も目立つスタイリッシュな体型！全体的に四角形のようなフレーム感があるのが特徴！',
                    ]
                }
                type_reference_ex = ' '.join(reference_ex.get(predicted_type, []))

                # 骨格タイプごとの参考URL（仮のURL）
                reference_urls = {
                    'straight': [
                        {
                            'url': 'https://www.stlady.jp/',
                            'description': 'ストレートタイプの専用通販サイト'
                        },
                        {
                            'url': 'https://zozo.jp/fashionnews/cbkmagazine/118628/?srsltid=AfmBOorvWfwZQ9kvtobQ14nyisc5QVNUUDf8JtKBCLvwzoe2yZGiO3-D',
                            'description': 'ストレートタイプにおすすめのアイテム'
                        },
                        {
                            'url': 'https://www.sixty-percent.com/articles/816',
                            'description': 'ストレートタイプの似合う服とコーディネート例'
                        }
                    ],
                    'wave': [
                        {
                            'url': 'https://www.waverry.jp/',
                            'description': 'ウェーブタイプの専用通販サイト'
                        },
                        {
                            'url': 'https://zozo.jp/fashionnews/folk/113108/?kid=399999990&t=w&utm_source=wear&utm_medium=pc&utm_campaign=wear_ranking',
                            'description': 'ウェーブタイプにおすすめのアイテム'
                        },
                        {
                            'url': 'https://www.sixty-percent.com/articles/1378',
                            'description': 'ウェーブタイプの似合う服とコーディネート例'
                        }
                    ],
                    'natural': [
                        {
                            'url': 'https://www.naturily.jp/',
                            'description': 'ナチュラルタイプの専用通販サイト'
                        },
                        {
                            'url': 'https://zozo.jp/fashionnews/folk/109819/',
                            'description': 'ナチュラルタイプにおすすめのアイテム'
                        },
                        {
                            'url': 'https://www.sixty-percent.com/articles/1222',
                            'description': 'ナチュラルタイプの似合う服とコーディネート例'
                        }
                    ]
                }
                type_reference_urls = reference_urls.get(predicted_type, [])


                # 結果の整形
                result = {
                    'predicted_type': predicted_type,
                    'type_based_recommendations': type_based_recommendations,
                    'uploaded_image': f'/static/uploads/{filename}',
                    'type_reference_urls': type_reference_urls,
                    'type_reference_ex': 
                    type_reference_ex
                }
                
                # 結果をセッションに保存してリダイレクト
                session['upload'] = result
                return redirect(url_for('show_results'))
                
            except FileNotFoundError:
                return jsonify({'error': 'モデルファイルが見つかりません。モデル学習を実行してください。'}), 500
            except ValueError as e:
                # MediaPipeで骨格が検出できない場合などに発生
                return jsonify({'error': f'画像処理エラー: {str(e)}'}), 500
            except Exception as e:
                # その他の予期せぬエラー
                traceback.print_exc()
                return jsonify({'error': f'分析中に予期せぬエラーが発生しました: {str(e)}'}), 500
        
        return jsonify({'error': '許可されていないファイル形式です'}), 400

@app.route('/results')
def show_results():
    """分析結果を表示するページ"""
    result = session.pop('upload', None) # セッションから結果を取得し削除

    if result is None:
        # セッションに結果がない場合（例: 直接/resultsにアクセス）はトップページに戻すかエラー表示
        return redirect('/')

    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)