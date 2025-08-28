#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルから推移律の矛盾を検出するスクリプト。

推移律の矛盾とは、3つのレコード(A, B, C)の関係が以下のような場合を指します:
- AとBの関係が True
- BとCの関係が True
- AとCの関係が False

このスクリプトは、グラフ理論を利用してこれらの「矛盾した三角形」を効率的に検出します。
"""

import pandas as pd
import networkx as nx
import argparse
import itertools
from tqdm import tqdm

def find_inconsistent_triangles(df, similarity_column):
    """
    DataFrameから推移律が矛盾した三角形を検出する。

    Args:
        df (pd.DataFrame): 評価データが含まれるDataFrame。
        similarity_column (str): 類似度(True/False)が含まれる列名。

    Returns:
        list: 矛盾した三角形のタプルのリスト。
    """
    # 1. すべてのペアの関係(True/False)を高速に参照できる辞書を作成
    all_pairs = {}
    for _, row in df.iterrows():
        # ペアのIDをソートして、(id1, id2)と(id2, id1)を同じキーで扱えるようにする
        pair = tuple(sorted((str(row['record_id_1']), str(row['record_id_2']))))
        all_pairs[pair] = bool(row[similarity_column])

    # 2. 関係がTrueのペアのみでグラフを構築
    G = nx.Graph()
    true_pairs_df = df[df[similarity_column] == True]
    for _, row in true_pairs_df.iterrows():
        G.add_edge(str(row['record_id_1']), str(row['record_id_2']))

    # 3. 矛盾した三角形を検出
    inconsistent_triangles = []
    nodes_to_check = [node for node in G.nodes() if G.degree(node) >= 2]

    print("矛盾した三角形を探索中...")
    for u in tqdm(nodes_to_check, desc="Checking nodes"):
        # あるノードuの隣人（uと関係がTrueのノード）をすべて取得
        neighbors = list(G.neighbors(u))
        
        # 隣人が2つ以上ある場合のみ、三角形を形成する可能性がある
        if len(neighbors) < 2:
            continue
        
        # 隣人同士のすべてのペアを確認
        for v, w in itertools.combinations(neighbors, 2):
            # この時点で (u, v) と (u, w) の関係はTrue
            # vとwの関係を確認する
            pair_vw = tuple(sorted((v, w)))
            
            # vとwの関係がFalse (またはCSVに存在しない) 場合、矛盾した三角形とみなす
            if not all_pairs.get(pair_vw, False):
                triangle = tuple(sorted((u, v, w)))
                inconsistent_triangles.append(triangle)
                
    # 重複して見つかった三角形を削除してユニークなリストを返す
    return list(set(inconsistent_triangles))

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="ペアワイズの類似度判定CSVから、推移律の矛盾を検出します。"
    )
    parser.add_argument(
        "--input-csv", 
        required=True, 
        help="入力CSVファイルへのパス。"
    )
    parser.add_argument(
        "--output-csv", 
        help="検出された矛盾した三角形を出力するCSVファイルへのパス（任意）。"
    )
    parser.add_argument(
        "--similarity-column", 
        default="predicted_similar_after", 
        help="類似度(True/False)が格納されている列の名前。"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {args.input_csv}")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に問題が発生しました: {e}")
        return
        
    # 必須列の存在チェック
    required_cols = ['record_id_1', 'record_id_2', args.similarity_column]
    if not all(col in df.columns for col in required_cols):
        print(f"エラー: CSVには次の列が必要です: {', '.join(required_cols)}")
        return
        
    # 類似度カラムをブール型に変換（"True", "False"のような文字列も正しく解釈）
    if df[args.similarity_column].dtype != bool:
        df[args.similarity_column] = df[args.similarity_column].apply(
            lambda x: str(x).strip().lower() in ['true', '1', 't', 'y', 'yes']
        )

    triangles = find_inconsistent_triangles(df, args.similarity_column)

    if not triangles:
        print("\n矛盾した三角形は見つかりませんでした。")
        return

    print(f"\n合計 {len(triangles)} 件の矛盾した三角形を検出しました。")

    # 結果の出力
    if args.output_csv:
        output_df = pd.DataFrame(
            triangles, columns=['record_id_A', 'record_id_B', 'record_id_C']
        )
        output_df.to_csv(args.output_csv, index=False)
        print(f"結果を {args.output_csv} に保存しました。")
    else:
        print("\n--- 検出された矛盾した三角形 (上位20件) ---")
        for t in triangles[:20]:
            print(t)
        if len(triangles) > 20:
            print(f"...他 {len(triangles) - 20} 件")

if __name__ == "__main__":
    main()