# 骨格タイプ判別システム

このプロジェクトは、MediaPipeを使用して骨格特徴量を抽出し、機械学習モデルを用いて骨格タイプ（ストレート、ウェーブ、ナチュラル）を判別するシステムです。

## 機能

- MediaPipeを使用した骨格特徴量の抽出
- データ拡張による学習データの増強
- 複数の機械学習モデル（Random Forest, SVM, XGBoost）による分類
- クロスバリデーションによるモデル評価
- モデルの保存と読み込み
- Flask Webアプリケーション
- Herokuデプロイ対応

## 必要条件

- Python 3.10以上
- 必要なパッケージは `requirements.txt` に記載

## インストール

1. リポジトリをクローン
```bash
git clone [repository-url]
cd [repository-name]
```

2. 必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

## ローカルでの実行

```bash
python app.py
```

ブラウザで `http://localhost:5000` にアクセスしてください。

## Herokuデプロイ

### 1. Heroku CLIのインストール

[Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)をインストールしてください。

### 2. Herokuアプリの作成

```bash
# Herokuにログイン
heroku login

# 新しいアプリを作成
heroku create your-app-name

# または既存のアプリを使用
heroku git:remote -a your-app-name
```

### 3. 環境変数の設定

```bash
# シークレットキーを設定
heroku config:set SECRET_KEY=your-secret-key-here
```

### 4. デプロイ

```bash
# 変更をコミット
git add .
git commit -m "Deploy to Heroku"

# Herokuにプッシュ
git push heroku main
```

### 5. アプリの起動

```bash
heroku open
```

### トラブルシューティング

デプロイでエラーが発生した場合：

1. ログを確認
```bash
heroku logs --tail
```

2. アプリの状態を確認
```bash
heroku ps
```

3. 必要に応じてアプリを再起動
```bash
heroku restart
```

## 使用方法

### 1. データの準備

画像データを以下のような構造で配置してください：

```
data/
├── straight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── wave/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── natural/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. データ拡張

```bash
python data_augmentation.py
```

### 3. モデルの学習

```bash
python skeleton_classifier.py
```

### 4. 新しい画像の判別

```python
from skeleton_classifier import SkeletonClassifier

# モデルの読み込み
classifier = SkeletonClassifier()
classifier.load_model('skeleton_classifier.joblib')

# 新しい画像の判別
result = classifier.predict('path/to/image.jpg')
print(f"判別結果: {result}")
```

## 特徴量

以下の特徴量を使用して骨格タイプを判別します：

- 肩幅とヒップ幅の比率
- ウエスト位置
- 胴体の縦横比
- 関節の突出度（肩、ヒップ）

## モデルの評価

- 訓練データとテストデータの分割比率: 8:2
- 5分割交差検証による評価
- 混同行列による誤分類パターンの分析

## 注意事項

- 画像は正面を向いた全身の写真を使用してください
- 骨格が正しく検出できない場合は、画像の品質や姿勢を確認してください
- データ拡張により生成された画像は、元の画像と同じ骨格タイプとして扱われます
- Herokuデプロイ時は、モデルファイルが大きい場合はGit LFSの使用を検討してください

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 