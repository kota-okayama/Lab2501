import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# llm_api_cache.json のパス (適宜修正してください)
cache_file_path = "siamese_model_pytorch/evaluation_results/llm_api_cache.json"
base_model_scores = []

with open(cache_file_path, "r", encoding="utf-8") as f:
    cache_data = json.load(f)

for key, value in cache_data.items():
    if "ft:" not in key:  # ベースモデルのキーを判定 (より厳密な判定が必要な場合もあります)
        if "score" in value:
            base_model_scores.append(value["score"])

if not base_model_scores:
    print("ベースモデルのスコアが見つかりませんでした。キーの判定条件を確認してください。")
else:
    print(f"抽出されたベースモデルのスコア数: {len(base_model_scores)}")

    # ヒストグラム
    plt.figure(figsize=(10, 6))
    plt.hist(base_model_scores, bins=50, edgecolor="black")
    plt.title("Base Model Score Distribution (Histogram)")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # KDEプロット
    plt.figure(figsize=(10, 6))
    sns.kdeplot(base_model_scores, fill=True)
    plt.title("Base Model Score Distribution (KDE)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    # 簡単な統計値
    print(f"スコアの平均値: {np.mean(base_model_scores):.4f}")
    print(f"スコアの中央値: {np.median(base_model_scores):.4f}")
    print(f"スコアの標準偏差: {np.std(base_model_scores):.4f}")
    print(f"スコアの25パーセンタイル: {np.percentile(base_model_scores, 25):.4f}")
    print(f"スコアの75パーセンタイル: {np.percentile(base_model_scores, 75):.4f}")
