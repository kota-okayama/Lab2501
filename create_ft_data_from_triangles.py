import pandas as pd
import yaml
import json
import argparse
import os
import sys
from pathlib import Path

# グローバル変数
BIB_DATA = {}

def load_bib_data(yaml_path):
    """YAMLファイルから書誌データを読み込む"""
    global BIB_DATA
    if not os.path.exists(yaml_path):
        print(f"エラー: YAMLファイルが見つかりません: {yaml_path}", file=sys.stderr)
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 'records' キーが存在する新しい形式のYAMLに対応
    if 'records' in data:
        for cluster_id, records_in_cluster in data['records'].items():
            for record in records_in_cluster:
                BIB_DATA[str(record['id'])] = record.get('data', {})
    else:
        print("警告: YAMLファイルに 'records' キーが見つかりませんでした。古い形式を試みます。", file=sys.stderr)
        # 必要であれば古い形式のローディングロジックをここに追加
        # 今回は新しい形式のみを想定
    
    if not BIB_DATA:
        print("エラー: YAMLからデータをロードできませんでした。ファイルの形式を確認してください。", file=sys.stderr)
        sys.exit(1)
    
    print(f"{len(BIB_DATA)}件のレコード情報をロードしました。")

def get_record_details_for_prompt(record_id):
    """レコードIDに対応する詳細情報を文字列として取得する"""
    record_details = BIB_DATA.get(str(record_id))
    if not record_details:
        return f"レコードID {record_id} の情報なし"
    return "\n".join([f"{key}: {value}" for key, value in record_details.items()])

def get_system_prompt(data_type):
    """データタイプに応じたシステムプロンプトを生成する"""
    prompt_map = {
        "bib": "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。",
        "music": "あなたは2つの楽曲情報が実質的に同一の楽曲を指すかどうかを判断する専門家です。",
        "person": "あなたは2つの人物情報が実質的に同一の人物を指すかどうかを判断する専門家です。"
    }
    base_prompt = prompt_map.get(data_type, "あなたは2つの情報が同一のものを指すか判断する専門家です。")
    instruction = (
        "まず、2つの情報が同一と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\n"
        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
    )
    return f"{base_prompt}\n{instruction}"

def create_finetuning_message(id1, id2, is_similar, data_type):
    """ファインチューニング用のメッセージオブジェクトを作成する"""
    system_prompt = get_system_prompt(data_type)
    
    info_name_map = {"bib": "書誌情報", "music": "楽曲情報", "person": "人物情報"}
    info_name = info_name_map.get(data_type, "情報")
    
    details1 = get_record_details_for_prompt(id1)
    details2 = get_record_details_for_prompt(id2)
    
    user_prompt = (
        f"以下の2つの{info_name}が、実質的に同一のものを指しているかどうかを判断してください。\n\n"
        f"{info_name}1:\n{details1}\n\n"
        f"{info_name}2:\n{details2}\n\n"
        "これらは同一のものですか？\n回答:"
    )
    
    assistant_response = "はい\n類似度スコア: 1.0" if is_similar else "いいえ\n類似度スコア: 0.0"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }

def main(args):
    """メイン処理"""
    load_bib_data(args.ground_truth_yaml)
    
    try:
        df = pd.read_csv(args.inconsistent_triangles_csv)
    except FileNotFoundError:
        print(f"エラー: 入力CSVファイルが見つかりません: {args.inconsistent_triangles_csv}", file=sys.stderr)
        sys.exit(1)

    finetuning_samples = []
    processed_pairs = set()

    print("CSVファイルからペアを抽出してファインチューニングデータを作成します...")
    for _, row in df.iterrows():
        nodes = {
            'n1': row['triangle_node1'],
            'n2': row['triangle_node2'],
            'n3': row['triangle_node3']
        }
        edges = {
            ('n1', 'n2'): row['true_edge12'],
            ('n2', 'n3'): row['true_edge23'],
            ('n1', 'n3'): row['true_edge31']
        }

        for (node_key1, node_key2), is_similar in edges.items():
            id1, id2 = str(nodes[node_key1]), str(nodes[node_key2])
            
            # 重複ペアをスキップ
            pair_key = tuple(sorted((id1, id2)))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            message = create_finetuning_message(id1, id2, is_similar, args.data_type)
            finetuning_samples.append(message)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in finetuning_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    print(f"\n完了: {len(finetuning_samples)}件のユニークなファインチューニング用サンプルを {output_path} に保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="矛盾する三角形が記述されたCSVからファインチューニング用のJSONLデータを生成します。"
    )
    parser.add_argument(
        "--inconsistent_triangles_csv", required=True,
        help="矛盾ペア情報が記載された入力CSVファイルのパス。"
    )
    parser.add_argument(
        "--ground_truth_yaml", required=True,
        help="レコードの詳細情報が含まれる正解データのYAMLファイルパス。"
    )
    parser.add_argument(
        "--output_jsonl", required=True,
        help="出力するJSONLファイルのパス。"
    )
    parser.add_argument(
        "--data_type",
        required=True,
        choices=["bib", "music", "person"],
        help="データの種類 (プロンプト生成に利用)。"
    )
    
    args = parser.parse_args()
    main(args) 