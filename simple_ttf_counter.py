#!/usr/bin/env python3
"""
シンプルなT,T,F矛盾三角形カウンタ

run_2k_{datatype}フォルダの*details.csvファイルから
predicted_similar_beforeを使用してT,T,Fパターンの矛盾三角形を直接カウントする。
"""

import pandas as pd
import os
import glob
from collections import defaultdict
from itertools import combinations
import re

def load_predictions(file_path):
    """
    CSVファイルから予測データを読み込み、辞書形式で返す
    
    Returns:
        dict: {(record_id_1, record_id_2): True/False}
    """
    print(f"Loading: {file_path}")
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
    
    print(f"  Loaded {len(predictions)} pairs")
    return predictions

def get_all_records(predictions):
    """
    全てのレコードIDを取得
    """
    records = set()
    for id1, id2 in predictions.keys():
        records.add(id1)
        records.add(id2)
    return list(records)

def count_ttf_triangles(predictions):
    """
    T,T,Fパターンの矛盾三角形をカウントする
    """
    records = get_all_records(predictions)
    ttf_count = 0
    ttf_triangles = []
    
    print(f"Checking triangles among {len(records)} records...")
    
    total_triangles = 0
    # 全ての3つ組み合わせをチェック
    for i, (a, b, c) in enumerate(combinations(records, 3)):
        total_triangles += 1
        if total_triangles % 100000 == 0:
            print(f"  Processed {total_triangles:,} triangles...")
        
        # 3つのペアの予測を取得
        ab_key = tuple(sorted([a, b]))
        bc_key = tuple(sorted([b, c]))
        ac_key = tuple(sorted([a, c]))
        
        # 全てのペアが存在するかチェック
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
    
    print(f"  Total triangles checked: {total_triangles:,}")
    print(f"  Found {ttf_count:,} T,T,F triangles")
    return ttf_count, ttf_triangles

def extract_datatype_from_path(file_path):
    """
    ファイルパスからデータタイプを抽出
    """
    # results_xxx/run_2k_xxx/ のパターンから抽出
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
        print(f"\n=== Processing {datatype} ===")
        
        try:
            # 予測データを読み込み
            predictions = load_predictions(file_path)
            
            if len(predictions) == 0:
                print("No predictions loaded, skipping...")
                results.append({
                    'datatype': datatype,
                    'file_path': file_path,
                    'total_pairs': 0,
                    'ttf_triangles': 0,
                    'error': 'No predictions loaded'
                })
                continue
            
            # T,T,F矛盾三角形をカウント
            ttf_count, triangles = count_ttf_triangles(predictions)
            
            results.append({
                'datatype': datatype,
                'file_path': file_path,
                'total_pairs': len(predictions),
                'ttf_triangles': ttf_count,
                'ttf_rate': ttf_count / len(predictions) if len(predictions) > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append({
                'datatype': datatype,
                'file_path': file_path,
                'total_pairs': 0,
                'ttf_triangles': 0,
                'error': str(e)
            })
    
    # 結果を表示
    print("\n" + "="*80)
    print("T,T,F矛盾三角形カウント結果")
    print("="*80)
    
    for result in results:
        print(f"\nデータタイプ: {result['datatype']}")
        print(f"ファイル: {os.path.basename(result['file_path'])}")
        print(f"総ペア数: {result['total_pairs']:,}")
        print(f"T,T,F矛盾三角形数: {result['ttf_triangles']:,}")
        if result['total_pairs'] > 0:
            print(f"T,T,F矛盾率: {result['ttf_rate']:.6f} ({result['ttf_rate']*100:.4f}%)")
        if 'error' in result:
            print(f"エラー: {result['error']}")
    
    # CSVファイルに保存
    results_df = pd.DataFrame(results)
    output_file = "ttf_triangles_count_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n結果をCSVファイルに保存しました: {output_file}")
    
    # サマリー
    total_triangles = sum(r['ttf_triangles'] for r in results if 'error' not in r)
    total_pairs = sum(r['total_pairs'] for r in results if 'error' not in r)
    
    print(f"\n=== サマリー ===")
    print(f"全データタイプ合計:")
    print(f"  総ペア数: {total_pairs:,}")
    print(f"  T,T,F矛盾三角形数: {total_triangles:,}")
    if total_pairs > 0:
        print(f"  全体T,T,F矛盾率: {total_triangles/total_pairs:.6f} ({total_triangles/total_pairs*100:.4f}%)")

if __name__ == "__main__":
    main()

