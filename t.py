import pandas as pd
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details_csv", required=True)
    parser.add_argument("--ground_truth_yaml", required=True)
    args = parser.parse_args()

    # 正解クラスタをロード
    record_to_cluster = {}
    with open(args.ground_truth_yaml, 'r', encoding='utf-8') as f:
        gt_data = yaml.safe_load(f)
    key = 'records' if 'records' in gt_data else 'clusters'
    for cid, records in gt_data[key].items():
        for record in records:
            rid = record.get('id') or record.get('record_id')
            record_to_cluster[str(rid)] = str(cid)

    # 評価詳細CSVをロード
    df = pd.read_csv(args.details_csv)
    
    # ground_truth_similar 列を再計算して検証
    df['is_truly_similar_check'] = df.apply(
        lambda row: record_to_cluster.get(str(row['record_id_1'])) == record_to_cluster.get(str(row['record_id_2'])),
        axis=1
    )

    total_pairs = len(df)
    positive_pairs = df['is_truly_similar_check'].sum()
    
    print(f"ファイル: {args.details_csv}")
    print(f"総ペア数: {total_pairs}")
    print(f"正例ペア数: {positive_pairs}")
    if total_pairs > 0:
        print(f"正例の割合: {(positive_pairs / total_pairs) * 100:.2f}%")

if __name__ == "__main__":
    main()