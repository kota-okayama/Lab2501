import pandas as pd
import networkx as nx
import yaml
import argparse
from collections import defaultdict

def load_yaml_data(gt_yaml_path):
    """YAMLから正解クラスタ情報と書誌データを読み込む"""
    with open(gt_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    gt_map = {}
    bib_data = {}
    key = 'records' if 'records' in data else 'clusters'
    for cluster_id, records in data[key].items():
        is_singleton = len(records) == 1
        for record in records:
            record_id = record.get('id') or record.get('record_id')
            if not is_singleton:
                gt_map[record_id] = str(cluster_id)
            
            if 'data' in record:
                bib_record = record['data'].copy()
            else:
                bib_record = record.copy()
            bib_data[record_id] = bib_record
    return gt_map, bib_data


def form_clusters_from_pairs(pairs, all_record_ids):
    """ペアリストからクラスタを形成する"""
    graph = nx.Graph()
    graph.add_nodes_from(all_record_ids)
    graph.add_edges_from(pairs)
    
    cluster_map = {}
    for i, component in enumerate(nx.connected_components(graph)):
        for record_id in component:
            cluster_map[record_id] = f"pred_{i}"
    return cluster_map

def analyze_pairwise_changes(df, bib_data):
    """ペアワイズ予測の変化を分析し、ファイルに保存する"""
    changed_df = df[df['predicted_similar_zeroshot'] != df['predicted_similar_fewshot']].copy()
    
    if changed_df.empty:
        print("ペアワイズ予測に変化はありませんでした。")
        return

    def get_title(record_id):
        """レコードIDから書誌タイトルを取得する"""
        return bib_data.get(str(record_id), {}).get('bib1_title', 'N/A')

    # タイトル情報を追加
    changed_df['title_1'] = changed_df['record_id_1'].apply(get_title)
    changed_df['title_2'] = changed_df['record_id_2'].apply(get_title)

    def get_change_type(row):
        gt = row['ground_truth_similar']
        zero = row['predicted_similar_zeroshot']
        few = row['predicted_similar_fewshot']
        if gt:  # Positive Pair
            if not zero and few: return "FN -> TP (改善)"
            if zero and not few: return "TP -> FN (悪化)"
        else:  # Negative Pair
            if not zero and few: return "TN -> FP (悪化)"
            if zero and not few: return "FP -> TN (改善)"
        return "その他"

    changed_df['change_type'] = changed_df.apply(get_change_type, axis=1)
    
    # 出力する列を整理
    output_columns = [
        'record_id_1', 'title_1', 'record_id_2', 'title_2', 'change_type',
        'ground_truth_similar', 'predicted_similar_zeroshot',
        'predicted_similar_fewshot', 'score_zeroshot', 'score_fewshot'
    ]
    output_df = changed_df[output_columns]

    output_filename = "changed_pairs_analysis.csv"
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n--- ペアワイズ変化の分析 ---")
    print(f"予測が変化したペア数: {len(changed_df)}件")
    print(f"変化の内訳:")
    print(changed_df['change_type'].value_counts())
    print(f"\n詳細を {output_filename} に保存しました。")

def analyze_clustering_changes(df, gt_map):
    """クラスタリングの変化を分析し、レポートを生成する"""
    all_record_ids = pd.concat([df['record_id_1'], df['record_id_2']]).unique()

    # Zero-shotとFew-shotのクラスタを再構築
    positive_pairs_zero = df[df['predicted_similar_zeroshot'] == True][['record_id_1', 'record_id_2']].values
    positive_pairs_few = df[df['predicted_similar_fewshot'] == True][['record_id_1', 'record_id_2']].values
    
    clusters_zero = form_clusters_from_pairs(positive_pairs_zero, all_record_ids)
    clusters_few = form_clusters_from_pairs(positive_pairs_few, all_record_ids)

    # 正解クラスタごとに、予測クラスタの構成を分析
    gt_to_pred_zero = defaultdict(set)
    gt_to_pred_few = defaultdict(set)
    
    # record_id -> gt_id の逆引きマップも作成
    record_to_gt = {rid: gid for rid, gid in gt_map.items()}

    for record_id in all_record_ids:
        gt_id = record_to_gt.get(record_id)
        if gt_id:
            if record_id in clusters_zero:
                gt_to_pred_zero[gt_id].add(clusters_zero[record_id])
            if record_id in clusters_few:
                gt_to_pred_few[gt_id].add(clusters_few[record_id])

    # レポート生成
    report_lines = ["--- クラスタリング変化の分析レポート ---"]
    degraded_clusters = 0
    
    for gt_id in sorted(gt_to_pred_zero.keys()):
        num_splits_zero = len(gt_to_pred_zero[gt_id])
        num_splits_few = len(gt_to_pred_few[gt_id])

        # Few-shotで分割数が増えた（悪化）したクラスタを報告
        if num_splits_few > num_splits_zero:
            degraded_clusters += 1
            report_lines.append(f"\n[悪化検出] 正解クラスタID: {gt_id}")
            report_lines.append(f"  - Zero-shot予測: {num_splits_zero}個に分割")
            report_lines.append(f"  - Few-shot予測 : {num_splits_few}個に分割")
    
    if not degraded_clusters:
         report_lines.append("\nクラスタリングが悪化した（分割数が増えた）正解クラスタは見つかりませんでした。")
    
    output_filename = "clustering_degradation_report.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"\n--- クラスタリング変化の分析 ---")
    print(f"クラスタリングが悪化した（分割数が増えた）正解クラスタ数: {degraded_clusters}件")
    print(f"詳細を {output_filename} に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="Zero-shotとFew-shotの評価結果の差異を分析するスクリプト")
    parser.add_argument("--details-csv", required=True, help="ペアごとの詳細な結果が記載されたCSVファイル")
    parser.add_argument("--gt-yaml", required=True, help="正解クラスタ情報が記載されたYAMLファイル")
    args = parser.parse_args()

    print(f"詳細結果ファイルを読み込み中: {args.details_csv}")
    try:
        df = pd.read_csv(args.details_csv)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {args.details_csv}")
        return
        
    print(f"正解YAMLファイルを読み込み中: {args.gt_yaml}")
    try:
        gt_map, bib_data = load_yaml_data(args.gt_yaml)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {args.gt_yaml}")
        return

    # 1. ペアワイズの変化を分析
    analyze_pairwise_changes(df, bib_data)

    # 2. クラスタリングの変化を分析
    analyze_clustering_changes(df, gt_map)

if __name__ == "__main__":
    main() 