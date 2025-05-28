import os

def list_files_in_directory(straight):
    try:
        # ディレクトリ内のファイル一覧を取得
        files = os.listdir(straight)
        # ディレクトリのみを除外してファイルのみをリストに格納
        straight_list = [file for file in files if os.path.isfile(os.path.join(straight, file))]
        return straight_list
    except Exception as e:
        print(f"Error: {e}")
        return []

# 使用例
straight = r'straight'
straight_list = list_files_in_directory(straight)
print(straight_list)

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pandas as pd
# Mediapipe Poseモジュールの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawingモジュールの初期化
mp_drawing = mp.solutions.drawing_utils

# データフレーム用のリスト
img_list = list()
land_index_list = list()
x_list = list()
y_list = list()
z_list = list()

# テスト用の画像の読み込み
for image_path in straight_list:
    # ここに画像のパスを指定します
    image = cv2.imread("straight/"+image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 画像の処理
    results = pose.process(image_rgb)

    # ランドマークを描画するための画像のコピー
    output_image = image.copy()
    
    # 全身のランドマークを描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            output_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
    
    # 画像の表示
    plt.figure(figsize=[10, 10])
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # 各ランドマークの座標データの取得と表示
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f'Landmark {idx}: (x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility})')
            img_list.append(image_path)
            land_index_list.append(idx)
            x_list.append(landmark.x)
            y_list.append(landmark.y)
            z_list.append(landmark.z)
# データフレームを作成
df = pd.DataFrame(
    {
        "画像名":img_list,
        "インデックス":land_index_list,
        "X":x_list,
        "Y":y_list,
        "Z":z_list
    }
)
print(df.head())
print(df.info())