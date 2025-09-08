#!/usr/bin/env python3
"""
グラフベースT,T,F矛盾三角形カウンタ

KNNグラフを利用して、実際にエッジが存在する三角形のみをチェックすることで
大幅に高速化したT,T,Fパターンの矛盾三角形カウンタ。
"""

import pandas as pd
import json
import os
import glob
from collections import defaultdict
import re

def load_knn_graph(graph_file):
    """
    KNNグラフファイルを読み込み、隣接リストを返す
    
    Returns:
        dict: {node_id: [neighbor_ids]}
    """
    print(f"Loading KNN graph: {graph_file}")
    
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
    
    print(f"  Loaded graph with {len(graph_data)} nodes")
    
    # エッジ数をカウント
    total_edges = sum(len(neighbors) for neighbors in graph_data.values())
    print(f"  Total directed edges: {total_edges}")
    
    return graph_data

def load_predictions(file_path):
    """
    CSVファイルから予測データを読み込み、辞書形式で返す
    
    Returns:
        dict: {(record_id_1, record_id_2): True/False}
    """
    print(f"Loading predictions: {file_path}")
    df = pd.read_csv(file_path)
    
    predictions = {}
    for _, row in df.iterrows():
        id1, id2 = row['record_id_1'], row['record_id_2']
        predicted = row['predicted_similar_before']
        
        # 文字列をブール値に変換
        if isinstance(predicted, str):
            predicted_bool = predicted.lower() == 'true'
        else:
            predicted_bool = bool(predicted)
        
        # 順序を統一（辞書順）
        key = tuple(sorted([id1, id2]))
        predictions[key] = predicted_bool
    
    print(f"  Loaded {len(predictions)} prediction pairs")
    return predictions

def find_triangles_in_graph(graph):
    """
    KNNグラフ内の全ての三角形を効率的に見つける
    
    Returns:
        list: [(node_a, node_b, node_c), ...]
    """
    print("Finding triangles in KNN graph...")
    triangles = []
    
    # 各ノードについて
    for u in graph:
        neighbors_u = set(graph[u])
        
        # uの隣接ノードのペアをチェック
        neighbors_list = list(neighbors_u)
        for i in range(len(neighbors_list)):
            for j in range(i + 1, len(neighbors_list)):
                v = neighbors_list[i]
                w = neighbors_list[j]
                
                # v-w間にもエッジがあるかチェック
                if w in graph.get(v, []) or v in graph.get(w, []):
                    # 三角形発見: u-v, u-w, v-w
                    triangle = tuple(sorted([u, v, w]))
                    triangles.append(triangle)
    
    # 重複を除去
    unique_triangles = list(set(triangles))
    print(f"  Found {len(unique_triangles)} unique triangles")
    
    return unique_triangles

def count_ttf_triangles_fast(triangles, predictions):
    """
    見つかった三角形からT,T,Fパターンをカウントする
    """
    print("Counting T,T,F patterns in triangles...")
    
    ttf_count = 0
    ttf_triangles = []
    
    for triangle in triangles:
        a, b, c = triangle
        
        # 3つのペアの予測を取得
        ab_key = tuple(sorted([a, b]))
        bc_key = tuple(sorted([b, c]))
        ac_key = tuple(sorted([a, c]))
        
        # 全てのペアが予測データに存在するかチェック
        if ab_key in predictions and bc_key in predictions and ac_key in predictions:
            ab_pred = predictions[ab_key]
            bc_pred = predictions[bc_key]
            ac_pred = predictions[ac_key]
            
            # T,T,Fパターンをチェック（順序無関係）
            predictions_list = [ab_pred, bc_pred, ac_pred]
            true_count = sum(predictions_list)
            
            if true_count == 2:  # 2つがTrue、1つがFalse
                ttf_count += 1
                ttf_triangles.append({
                    'record_a': a,
                    'record_b': b, 
                    'record_c': c,
                    'ab_prediction': ab_pred,
                    'bc_prediction': bc_pred,
                    'ac_prediction': ac_pred
                })
    
    print(f"  Found {ttf_count} T,T,F triangles out of {len(triangles)} total triangles")
    return ttf_count, ttf_triangles

def find_graph_file(datatype_dir):
    """
    データタイプディレクトリからKNNグラフファイルを探す
    """
    graph_patterns = [
        f"{datatype_dir}/graphs/knn_graph_full_k*.json",
        f"{datatype_dir}/graphs/merged_knn_graph_k*.json",
        f"{datatype_dir}/graphs/*knn*.json"
    ]
    
    for pattern in graph_patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]  # 最初に見つかったファイルを使用
    
    return None

def extract_datatype_from_path(file_path):
    """
    ファイルパスからデータタイプを抽出
    """
    match = re.search(r'results_([^/]+)/run_2k_', file_path)
    if match:
        return match.group(1)
    return "unknown"

def main():
    # 全ての対象ファイルを検索
    file_pattern = "results_*/run_2k_*/evaluation_results/*details.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        print("No details.csv files found!")
        return
    
    print(f"Found {len(files)} details.csv files:")
    for f in files:
        print(f"  {f}")
    print()
    
    results = []
    
    for file_path in files:
        datatype = extract_datatype_from_path(file_path)
        datatype_dir = os.path.dirname(os.path.dirname(file_path))  # run_2k_xxx ディレクトリ
        
        print(f"\n=== Processing {datatype} ===")
        print(f"CSV file: {file_path}")
        print(f"Directory: {datatype_dir}")
        
        try:
            # KNNグラフファイルを探す
            graph_file = find_graph_file(datatype_dir)
            if not graph_file:
                print(f"No KNN graph file found in {datatype_dir}")
                results.append({
                    'datatype': datatype,
                    'file_path': file_path,
                    'total_pairs': 0,
                    'total_triangles': 0,
                    'ttf_triangles': 0,
                    'error': 'No KNN graph file found'
                })
                continue
            
            print(f"Using graph file: {graph_file}")
            
            # KNNグラフを読み込み
            graph = load_knn_graph(graph_file)
            
            # 予測データを読み込み
            predictions = load_predictions(file_path)
            
            if len(predictions) == 0:
                print("No predictions loaded, skipping...")
                results.append({
                    'datatype': datatype,
                    'file_path': file_path,
                    'total_pairs': 0,
                    'total_triangles': 0,
                    'ttf_triangles': 0,
                    'error': 'No predictions loaded'
                })
                continue
            
            # グラフ内の三角形を見つける
            triangles = find_triangles_in_graph(graph)
            
            # T,T,F矛盾三角形をカウント
            ttf_count, ttf_details = count_ttf_triangles_fast(triangles, predictions)
            
            results.append({
                'datatype': datatype,
                'file_path': file_path,
                'graph_file': graph_file,
                'total_pairs': len(predictions),
                'total_triangles': len(triangles),
                'ttf_triangles': ttf_count,
                'ttf_rate': ttf_count / len(triangles) if len(triangles) > 0 else 0,
                'ttf_vs_pairs_rate': ttf_count / len(predictions) if len(predictions) > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'datatype': datatype,
                'file_path': file_path,
                'total_pairs': 0,
                'total_triangles': 0,
                'ttf_triangles': 0,
                'error': str(e)
            })
    
    # 結果を表示
    print("\n" + "="*80)
    print("グラフベースT,T,F矛盾三角形カウント結果")
    print("="*80)
    
    for result in results:
        print(f"\nデータタイプ: {result['datatype']}")
        print(f"ファイル: {os.path.basename(result['file_path'])}")
        if 'graph_file' in result:
            print(f"グラフファイル: {os.path.basename(result['graph_file'])}")
        print(f"総ペア数: {result['total_pairs']:,}")
        print(f"総三角形数: {result['total_triangles']:,}")
        print(f"T,T,F矛盾三角形数: {result['ttf_triangles']:,}")
        if result['total_triangles'] > 0:
            print(f"三角形中のT,T,F率: {result['ttf_rate']:.6f} ({result['ttf_rate']*100:.4f}%)")
        if result['total_pairs'] > 0:
            print(f"ペア数に対するT,T,F率: {result['ttf_vs_pairs_rate']:.6f} ({result['ttf_vs_pairs_rate']*100:.4f}%)")
        if 'error' in result:
            print(f"エラー: {result['error']}")
    
    # CSVファイルに保存
    results_df = pd.DataFrame(results)
    output_file = "graph_based_ttf_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n結果をCSVファイルに保存しました: {output_file}")
    
    # サマリー
    successful_results = [r for r in results if 'error' not in r]
    total_triangles = sum(r['total_triangles'] for r in successful_results)
    total_ttf = sum(r['ttf_triangles'] for r in successful_results)
    total_pairs = sum(r['total_pairs'] for r in successful_results)
    
    print(f"\n=== サマリー ===")
    print(f"成功したデータセット: {len(successful_results)}/{len(results)}")
    print(f"全データタイプ合計:")
    print(f"  総ペア数: {total_pairs:,}")
    print(f"  総三角形数: {total_triangles:,}")
    print(f"  T,T,F矛盾三角形数: {total_ttf:,}")
    if total_triangles > 0:
        print(f"  全体T,T,F率（対三角形）: {total_ttf/total_triangles:.6f} ({total_ttf/total_triangles*100:.4f}%)")
    if total_pairs > 0:
        print(f"  全体T,T,F率（対ペア数）: {total_ttf/total_pairs:.6f} ({total_ttf/total_pairs*100:.4f}%)")

if __name__ == "__main__":
    main()
