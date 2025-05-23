import faiss
import numpy as np
import pickle
import os
import json
import time
import sys
import argparse

# プロジェクトルートをPythonパスに追加 (他モジュールをインポートするため)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def build_faiss_knn_graph(args):
    print(f"Starting K-NN graph construction with K={args.k_neighbors} using Faiss...")
    start_time = time.time()

    # --- データのロード ---
    print("\nStep 1: Loading record embeddings and IDs...")
    if not os.path.exists(args.embeddings_path) or not os.path.exists(args.ids_path):
        print(f"Error: Embeddings file ({args.embeddings_path}) or IDs file ({args.ids_path}) not found.")
        print("Please ensure these files exist, e.g., by running vectorize_records.py first.")
        return

    try:
        record_embeddings = np.load(args.embeddings_path)
        with open(args.ids_path, "rb") as f:
            record_ids_ordered = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if record_embeddings.ndim == 1:
        if record_embeddings.shape[0] > 0:
            record_embeddings = record_embeddings.reshape(1, -1)
        else:
            print("Error: Embeddings array is empty or malformed after loading.")
            return
    elif record_embeddings.ndim != 2:
        print(f"Error: Embeddings array has unexpected dimension: {record_embeddings.ndim}. Expected 2D array.")
        return

    num_records, dimension = record_embeddings.shape
    print(f"Loaded {num_records} records with embedding dimension {dimension}.")
    if len(record_ids_ordered) != num_records:
        print("Error: Number of embeddings does not match number of IDs.")
        return

    # Kの値を調整 (元々のKは引数で渡される k_neighbors)
    actual_k = args.k_neighbors
    if actual_k >= num_records:
        print(
            f"Warning: K ({actual_k}) is greater than or equal to the number of records ({num_records}). "
            f"Adjusting K to {num_records -1} for search (if num_records > 0)."
        )
        actual_k = num_records - 1 if num_records > 0 else 0

    if actual_k <= 0 and num_records > 0:  # レコードが1つしかない場合など actual_k が0以下になるケース
        print("Warning: K is 0 or less after adjustment (e.g. only one record exists). Cannot build KNN graph.")
        # 空のグラフを保存するか、ここで終了するか。今回は空のグラフを生成する方向に進める。
        # もし num_records == 1 の場合、num_neighbors_to_search は 1 になる。
        # index.search は結果を出すが、後続のループで自分自身は除外されるので空になる。

    # --- Faissインデックスの構築 ---
    print("\nStep 2: Building Faiss index...")
    try:
        index = faiss.IndexFlatL2(dimension)
        index.add(record_embeddings)
        print(f"Faiss index built. Total vectors in index: {index.ntotal}")
    except Exception as e:
        print(f"Error building Faiss index: {e}")
        print("Please ensure faiss-cpu or faiss-gpu is installed correctly.")
        return

    # --- K近傍探索の実行 ---
    # K+1 で検索 (自身を含むため)。
    # num_records が1の場合、num_neighbors_to_search は 1 となる (actual_k が0でも)
    num_neighbors_to_search = min(actual_k + 1, num_records) if num_records > 0 else 0

    knn_graph = {}  # 先に初期化

    if num_records == 0 or actual_k <= 0:  # レコードがない、またはKが0以下で探索不要な場合
        print(f"No records to process or K is {actual_k}. Skipping Faiss search. An empty graph will be saved.")
    else:
        print(
            f"\nStep 3: Searching for {actual_k} nearest neighbors (requesting {num_neighbors_to_search} from Faiss)..."
        )
        distances, indices = index.search(record_embeddings, num_neighbors_to_search)
        print("Search completed.")

        # --- 結果の処理とグラフ構築 ---
        print("\nStep 4: Processing search results and building graph...")
        for i in range(num_records):
            source_record_id = str(record_ids_ordered[i])  # IDを文字列に統一
            neighbor_ids_for_source = []
            if num_neighbors_to_search > 0:  # 探索が行われた場合のみ
                for j in range(indices.shape[1]):  # 実際に返ってきた近傍の数だけループ
                    neighbor_original_index = indices[i][j]
                    if neighbor_original_index == i:
                        continue
                    if neighbor_original_index == -1:  # 通常 IndexFlatL2 では発生しにくい
                        continue

                    # record_ids_ordered のインデックス範囲チェック
                    if 0 <= neighbor_original_index < len(record_ids_ordered):
                        neighbor_ids_for_source.append(
                            str(record_ids_ordered[neighbor_original_index])
                        )  # IDを文字列に統一
                    else:
                        print(
                            f"Warning: Invalid neighbor index {neighbor_original_index} for record {source_record_id}. Skipping this neighbor."
                        )

                    if len(neighbor_ids_for_source) == actual_k:
                        break
            knn_graph[source_record_id] = neighbor_ids_for_source
        print(f"K-NN graph constructed with {len(knn_graph)} nodes.")

    # --- グラフの保存 ---
    print("\nStep 5: Saving K-NN graph...")
    output_dir = os.path.dirname(args.output_graph_path)
    if output_dir:  # output_graph_path がファイル名のみでディレクトリなしの場合、output_dirは空文字列
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output_graph_path, "w", encoding="utf-8") as f:
            json.dump(knn_graph, f, indent=4, ensure_ascii=False)
        print(f"K-NN graph saved to {args.output_graph_path}")
    except Exception as e:
        print(f"Error saving graph to JSON: {e}")

    total_time = time.time() - start_time
    print(f"\nK-NN graph construction finished in {total_time // 60:.0f}m {total_time % 60:.0f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a K-NN graph using Faiss from record embeddings.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the record embeddings file (.npy).")
    parser.add_argument("--ids_path", type=str, required=True, help="Path to the record IDs file (.pkl).")
    parser.add_argument("--output_graph_path", type=str, required=True, help="Path to save the K-NN graph (.json).")
    parser.add_argument("--k_neighbors", type=int, default=10, help="Number of nearest neighbors (K).")

    cli_args = parser.parse_args()
    build_faiss_knn_graph(cli_args)
