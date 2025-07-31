import faiss
import numpy as np
import pickle
import os
import json
import time
import sys
import argparse
from collections import defaultdict

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def build_single_knn_graph(embeddings_path, ids_path, k_neighbors):
    """単一のエンベディングファイルからK近傍グラフを構築"""
    print(f"  エンベディングファイル: {embeddings_path}")
    
    # データのロード
    if not os.path.exists(embeddings_path) or not os.path.exists(ids_path):
        print(f"    エラー: ファイルが見つかりません")
        return {}

    try:
        record_embeddings = np.load(embeddings_path)
        with open(ids_path, "rb") as f:
            record_ids_ordered = pickle.load(f)
    except Exception as e:
        print(f"    エラー: データの読み込みに失敗: {e}")
        return {}

    if record_embeddings.ndim == 1:
        if record_embeddings.shape[0] > 0:
            record_embeddings = record_embeddings.reshape(1, -1)
        else:
            print("    エラー: エンベディング配列が空です")
            return {}
    elif record_embeddings.ndim != 2:
        print(f"    エラー: エンベディング配列の次元が不正: {record_embeddings.ndim}")
        return {}

    num_records, dimension = record_embeddings.shape
    print(f"    レコード数: {num_records}, 次元数: {dimension}")
    
    if len(record_ids_ordered) != num_records:
        print("    エラー: エンベディング数とID数が一致しません")
        return {}

    # Kの値を調整
    actual_k = min(k_neighbors, num_records - 1) if num_records > 1 else 0
    
    if actual_k <= 0:
        print("    警告: K=0以下のため、空のグラフを返します")
        return {str(record_id): [] for record_id in record_ids_ordered}

    # Faissインデックスの構築
    try:
        index = faiss.IndexFlatL2(dimension)
        index.add(record_embeddings)
    except Exception as e:
        print(f"    エラー: Faissインデックスの構築に失敗: {e}")
        return {}

    # K近傍探索
    num_neighbors_to_search = actual_k + 1  # 自分自身を含むため+1
    distances, indices = index.search(record_embeddings, num_neighbors_to_search)

    # グラフの構築
    knn_graph = {}
    for i in range(num_records):
        source_record_id = str(record_ids_ordered[i])
        neighbor_ids = []
        
        for j in range(indices.shape[1]):
            neighbor_idx = indices[i][j]
            if neighbor_idx == i:  # 自分自身をスキップ
                continue
            if neighbor_idx == -1:
                continue
            if 0 <= neighbor_idx < len(record_ids_ordered):
                neighbor_ids.append(str(record_ids_ordered[neighbor_idx]))
            if len(neighbor_ids) == actual_k:
                break
        
        knn_graph[source_record_id] = neighbor_ids

    print(f"    グラフ構築完了: {len(knn_graph)}ノード")
    return knn_graph


def merge_knn_graphs(graph_list, merge_method="union"):
    """複数のK近傍グラフを統合"""
    print(f"\nグラフの統合を開始 (方法: {merge_method})")
    
    if not graph_list:
        print("  統合対象のグラフがありません")
        return {}
    
    merged_graph = defaultdict(set)
    
    # 全グラフのエッジを収集
    total_edges = 0
    for i, graph in enumerate(graph_list):
        if not graph:
            continue
        print(f"  グラフ {i+1}: {len(graph)}ノード")
        for source_id, neighbors in graph.items():
            for neighbor_id in neighbors:
                merged_graph[source_id].add(neighbor_id)
                total_edges += 1
    
    # セットをリストに変換
    final_graph = {node_id: list(neighbors) 
                   for node_id, neighbors in merged_graph.items()}
    
    final_edges = sum(len(neighbors) for neighbors in final_graph.values())
    print(f"  統合前の総エッジ数: {total_edges}")
    print(f"  統合後のエッジ数: {final_edges}")
    print(f"  統合後のノード数: {len(final_graph)}")
    
    return final_graph


def main():
    parser = argparse.ArgumentParser(
        description="Build K-NN graphs from multiple embeddings and merge them."
    )
    parser.add_argument(
        "--embedding_summary_path", type=str, required=True,
        help="Path to the embedding summary JSON file."
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=10,
        help="Number of nearest neighbors (K)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save individual and merged graphs."
    )
    parser.add_argument(
        "--selected_combinations", type=str, default="",
        help="Comma-separated list of combinations to use for merging. "
             "If empty, all combinations will be used."
    )

    args = parser.parse_args()

    # サマリーファイルの読み込み
    print(f"エンベディングサマリーの読み込み: {args.embedding_summary_path}")
    if not os.path.exists(args.embedding_summary_path):
        print(f"エラー: サマリーファイルが見つかりません: {args.embedding_summary_path}")
        return

    with open(args.embedding_summary_path, "r", encoding="utf-8") as f:
        embedding_summary = json.load(f)

    print(f"読み込み成功: {len(embedding_summary)}種類のエンベディング")

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 処理対象の決定
    if args.selected_combinations:
        selected_names = [name.strip() for name in args.selected_combinations.split(",")]
        combinations_to_process = [
            item for item in embedding_summary 
            if item["name"] in selected_names
        ]
        print(f"選択された組み合わせ: {selected_names}")
    else:
        combinations_to_process = embedding_summary
        print("全ての組み合わせを処理対象とします")

    if not combinations_to_process:
        print("処理対象の組み合わせがありません")
        return

    # 各エンベディングからK近傍グラフを構築
    individual_graphs = {}
    graph_list_for_merge = []

    print(f"\n=== 個別グラフの構築 (K={args.k_neighbors}) ===")
    for item in combinations_to_process:
        combination_name = item["name"]
        embeddings_file = item["embeddings_file"]
        ids_file = item["ids_file"]
        
        print(f"\n{combination_name}のグラフを構築中...")
        graph = build_single_knn_graph(embeddings_file, ids_file, args.k_neighbors)
        
        if graph:
            individual_graphs[combination_name] = graph
            graph_list_for_merge.append(graph)
            
            # 個別グラフの保存
            individual_graph_path = os.path.join(
                args.output_dir, f"knn_graph_{combination_name}_k{args.k_neighbors}.json"
            )
            with open(individual_graph_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
            print(f"    保存: {individual_graph_path}")

    # グラフの統合
    if len(graph_list_for_merge) > 1:
        print("\n=== グラフの統合 ===")
        merged_graph = merge_knn_graphs(graph_list_for_merge, merge_method="union")
        
        # 統合グラフの保存
        merged_graph_path = os.path.join(
            args.output_dir, f"merged_knn_graph_k{args.k_neighbors}.json"
        )
        with open(merged_graph_path, "w", encoding="utf-8") as f:
            json.dump(merged_graph, f, indent=2, ensure_ascii=False)
        print(f"統合グラフを保存: {merged_graph_path}")
        
        # 統合結果のサマリー
        combination_names = [item["name"] for item in combinations_to_process]
        merge_summary = {
            "k_neighbors": args.k_neighbors,
            "source_combinations": combination_names,
            "individual_graphs": {
                name: {
                    "nodes": len(graph),
                    "edges": sum(len(neighbors) for neighbors in graph.values())
                } for name, graph in individual_graphs.items()
            },
            "merged_graph": {
                "nodes": len(merged_graph),
                "edges": sum(len(neighbors) for neighbors in merged_graph.values())
            },
            "merged_graph_file": merged_graph_path
        }
        
        summary_path = os.path.join(args.output_dir, f"merge_summary_k{args.k_neighbors}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(merge_summary, f, indent=2, ensure_ascii=False)
        print(f"統合サマリーを保存: {summary_path}")
        
    else:
        print(f"\n統合可能なグラフが{len(graph_list_for_merge)}個のため、統合をスキップします")
        # グラフが1つの場合、それをmerged_graphとして扱う
        if len(graph_list_for_merge) == 1:
            single_graph = graph_list_for_merge[0]
            merged_graph_path = os.path.join(
                args.output_dir, f"merged_knn_graph_k{args.k_neighbors}.json"
            )
            with open(merged_graph_path, "w", encoding="utf-8") as f:
                json.dump(single_graph, f, indent=2, ensure_ascii=False)
            print(f"単一グラフを統合済みグラフとして保存: {merged_graph_path}")

    print("\n=== 処理完了 ===")
    print(f"個別グラフ数: {len(individual_graphs)}")
    if len(graph_list_for_merge) > 1:
        print(f"統合グラフ: {merged_graph_path}")


if __name__ == "__main__":
    main() 