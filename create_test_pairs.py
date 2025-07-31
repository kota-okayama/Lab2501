import numpy as np
import pickle
import argparse
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def create_knn_pairs(embeddings_path, ids_path, output_csv, k_neighbors):
    """
    エンベディングからk-NNペアを生成し、CSVに出力する。
    """
    print(f"エンベディングを読み込み中: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    print(f"IDリストを読み込み中: {ids_path}")
    with open(ids_path, 'rb') as f:
        record_ids = pickle.load(f)
    
    if embeddings.shape[0] != len(record_ids):
        raise ValueError("エンベディングとIDリストの数が一致しません。")

    print(f"{k_neighbors}-NNモデルを構築中...")
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine', algorithm='brute')
    nn_model.fit(embeddings)
    
    print("近傍ペアを探索中...")
    distances, indices = nn_model.kneighbors(embeddings)
    
    pairs = set()
    for i in range(len(record_ids)):
        record_id_1 = record_ids[i]
        for j in range(1, indices.shape[1]): # 0番目は自分自身なのでスキップ
            neighbor_index = indices[i, j]
            record_id_2 = record_ids[neighbor_index]
            
            # ペアの順序を統一して重複を防ぐ
            sorted_pair = tuple(sorted((record_id_1, record_id_2)))
            pairs.add(sorted_pair)
            
    print(f"生成されたユニークペア数: {len(pairs)}")
    
    # DataFrameにしてCSV保存
    pairs_df = pd.DataFrame(list(pairs), columns=["record_id_1", "record_id_2"])
    pairs_df.to_csv(output_csv, index=False)
    
    print(f"評価ペアをCSVファイルに保存しました: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="エンベディングからk-NN評価ペアを生成するスクリプト。")
    parser.add_argument("--embeddings_path", required=True, help="エンベディングNPYファイルのパス。")
    parser.add_argument("--ids_path", required=True, help="レコードID Pickleファイルのパス。")
    parser.add_argument("--output_csv", required=True, help="出力するCSVファイルのパス。")
    parser.add_argument("--k_neighbors", type=int, default=10, help="各レコードに対して見つける近傍の数。")
    
    args = parser.parse_args()
    
    create_knn_pairs(args.embeddings_path, args.ids_path, args.output_csv, args.k_neighbors)

if __name__ == "__main__":
    main() 