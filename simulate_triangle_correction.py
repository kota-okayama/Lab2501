#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矛盾した三角形を完全に修正した場合のF1スコア改善をシミュレートするスクリプト。
"""

import pandas as pd
import argparse
import itertools
import yaml
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


def load_ground_truth_clusters(yaml_path):
    """YAMLファイルから正解クラスタ情報を読み込む"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"エラー: 正解データファイルが見つかりません: {yaml_path}")
        return None, None
    except Exception as e:
        print(f"エラー: YAMLファイルの読み込みに失敗しました: {e}")
        return None, None
    
    ground_truth_clusters = {}
    all_record_ids = set()
    
    # 新旧両方のYAML形式に対応
    records_key = 'records' if 'records' in data else 'clusters'
    if records_key not in data:
        print("エラー: YAMLに'records'または'clusters'キーがありません。")
        return None, None
        
    for cluster_id, records in data[records_key].items():
        for record in records:
            record_id = str(record.get('id') or record.get('record_id'))
            ground_truth_clusters[record_id] = str(cluster_id)
            all_record_ids.add(record_id)
            
    return ground_truth_clusters, sorted(list(all_record_ids))


def calculate_metrics(y_true, y_pred, model_name=""):
    """
    正解ラベルと予測ラベルから評価指標を計算し、表示する。
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n--- {model_name} 評価指標 ---")
    print(f"  +----------------+-----------------+-----------------+")
    print(f"  | {'':^14} | {'Predicted: Pos':^15} | {'Predicted: Neg':^15} |")
    print(f"  +================+=================+=================+")
    print(f"  | {'Actual: Pos':<14} | TP: {tp:<12d} | FN: {fn:<12d} |")
    print(f"  +----------------+-----------------+-----------------+")
    print(f"  | {'Actual: Neg':<14} | FP: {fp:<12d} | TN: {tn:<12d} |")
    print(f"  +----------------+-----------------+-----------------+")
    print(f"  適合率 (Precision): {precision:.4f}")
    print(f"  再現率 (Recall):    {recall:.4f}")
    print(f"  F1スコア:           {f1:.4f}")
    
    return {"f1": f1, "precision": precision, "recall": recall}

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="矛盾した三角形の修正によるF1スコア改善をシミュレートします。"
    )
    parser.add_argument(
        "--evaluation-csv",
        required=True,
        help="元の評価結果詳細CSVファイル (`..._details.csv`)。"
    )
    parser.add_argument(
        "--triangles-csv",
        required=True,
        help="`detect_inconsistent_triangles.py` が出力した矛盾三角形のCSVファイル。"
    )
    parser.add_argument(
        "--prediction-column",
        default="predicted_similar_after",
        help="修正対象の予測結果が含まれる列名。"
    )
    parser.add_argument(
        "--ground-truth-yaml",
        required=True,
        help="データセット全体の正解クラスタ情報を含むYAMLファイル。"
    )
    args = parser.parse_args()

    # --- データの読み込み ---
    try:
        eval_df = pd.read_csv(args.evaluation_csv)
        triangles_df = pd.read_csv(args.triangles_csv)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e.filename}")
        return
    except Exception as e:
        print(f"エラー: ファイル読み込み中に問題が発生しました: {e}")
        return

    ground_truth_clusters, all_record_ids = load_ground_truth_clusters(
        args.ground_truth_yaml
    )
    if not ground_truth_clusters:
        return

    # --- 矛盾に関わるペアの特定 ---
    triangle_pairs = set()
    for _, row in triangles_df.iterrows():
        ids = sorted([str(row['record_id_A']), str(row['record_id_B']), str(row['record_id_C'])])
        triangle_pairs.add(tuple(sorted((ids[0], ids[1]))))
        triangle_pairs.add(tuple(sorted((ids[0], ids[2]))))
        triangle_pairs.add(tuple(sorted((ids[1], ids[2]))))
    
    print(f"矛盾した三角形の数: {len(triangles_df)}")
    print(f"矛盾に関わるユニークなペアの数: {len(triangle_pairs)}")

    # --- シミュレーションの実行 ---
    # 新しい予測列を元の予測列で初期化
    simulated_col_name = f"simulated_{args.prediction_column}"
    eval_df[simulated_col_name] = eval_df[args.prediction_column]

    # 矛盾ペアの予測を正解ラベルで上書き
    corrected_count = 0
    for index, row in eval_df.iterrows():
        pair = tuple(sorted((str(row['record_id_1']), str(row['record_id_2']))))
        if pair in triangle_pairs:
            # 予測が間違っていた場合のみ修正し、カウントする
            if row[args.prediction_column] != row['ground_truth_similar']:
                corrected_count += 1
            eval_df.loc[index, simulated_col_name] = row['ground_truth_similar']
    
    print(f"シミュレーションで修正された予測の数: {corrected_count} 件")

    # --- 結果の比較 ---
    y_true = eval_df['ground_truth_similar']
    y_pred_before = eval_df[args.prediction_column]
    y_pred_after = eval_df[simulated_col_name]

    # 修正前のスコア
    metrics_before = calculate_metrics(y_true, y_pred_before, "修正前 (Original)")

    # シミュレーション後のスコア
    metrics_after = calculate_metrics(y_true, y_pred_after, "シミュレーション後 (Corrected)")

    # 改善度の表示
    f1_improvement = metrics_after['f1'] - metrics_before['f1']
    print("\n--- シミュレーション結果 ---")
    print(f"F1スコア改善の理論上の最大値: {f1_improvement:+.4f}")
    print(f"  ( {metrics_before['f1']:.4f} -> {metrics_after['f1']:.4f} )")

    # --- 全ペア推論のシミュレーション ---
    print("\n--- 全ペア推論シミュレーション ---")
    
    # 修正前の予測ペアでクラスタリング
    pred_clusters_before = form_predicted_clusters(
        eval_df, args.prediction_column, all_record_ids
    )
    
    # シミュレーション後の予測ペアでクラスタリング
    pred_clusters_after = form_predicted_clusters(
        eval_df, simulated_col_name, all_record_ids
    )
    
    # 全ペアのラベルを生成
    true_labels, pred_labels_before, pred_labels_after = generate_all_pairs_labels(
        all_record_ids, ground_truth_clusters, pred_clusters_before, pred_clusters_after
    )
    
    # 全ペア推論の評価
    print("\n[全ペア推論]")
    metrics_all_before = calculate_metrics(
        true_labels, pred_labels_before, "修正前 (Original - All-Pairs)"
    )
    metrics_all_after = calculate_metrics(
        true_labels, pred_labels_after, "シミュレーション後 (Corrected - All-Pairs)"
    )

    f1_improvement_all = metrics_all_after['f1'] - metrics_all_before['f1']
    print("\n--- 全ペア推論シミュレーション結果 ---")
    print(f"全ペアF1スコア改善の理論上の最大値: {f1_improvement_all:+.4f}")
    print(f"  ( {metrics_all_before['f1']:.4f} -> {metrics_all_after['f1']:.4f} )")


def form_predicted_clusters(df, pred_column, all_record_ids):
    """予測結果からクラスタを形成する"""
    graph = nx.Graph()
    graph.add_nodes_from(all_record_ids)
    
    true_pairs = df[df[pred_column] == True]
    edges = [
        (str(row['record_id_1']), str(row['record_id_2']))
        for _, row in true_pairs.iterrows()
    ]
    graph.add_edges_from(edges)
    
    predicted_cluster_map = {}
    for i, component in enumerate(nx.connected_components(graph)):
        for node in component:
            predicted_cluster_map[node] = i
            
    return predicted_cluster_map

def generate_all_pairs_labels(
    all_record_ids, true_clusters, pred_clusters_before, pred_clusters_after
):
    """データセットの全ペアについて、正解・予測ラベルを生成する"""
    true_labels = []
    pred_labels_before = []
    pred_labels_after = []
    
    print("全ペアのラベルを生成中...")
    for id1, id2 in tqdm(itertools.combinations(all_record_ids, 2), desc="Generating all-pairs labels"):
        # 正解ラベル
        is_true_similar = true_clusters.get(id1) == true_clusters.get(id2)
        true_labels.append(is_true_similar)
        
        # 修正前の予測ラベル
        c1_before = pred_clusters_before.get(id1, -1)
        c2_before = pred_clusters_before.get(id2, -2)
        pred_before_similar = (c1_before == c2_before)
        pred_labels_before.append(pred_before_similar)
        
        # シミュレーション後の予測ラベル
        c1_after = pred_clusters_after.get(id1, -1)
        c2_after = pred_clusters_after.get(id2, -2)
        pred_after_similar = (c1_after == c2_after)
        pred_labels_after.append(pred_after_similar)
        
    return true_labels, pred_labels_before, pred_labels_after


if __name__ == "__main__":
    main()