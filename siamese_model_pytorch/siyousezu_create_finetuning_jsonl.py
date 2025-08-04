import pandas as pd
import json
import os
import argparse
import yaml
import numpy as np

BIB_DATA = {}

def get_record_details_for_prompt(record_id):
    """指定されたレコードIDの書誌情報を整形して返す"""
    if not BIB_DATA:
        return f"書誌情報データ未ロード (ID: {record_id})"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    title = bib_details.get("title", bib_details.get("bib1_title", "タイトル不明"))
    authors_list = bib_details.get("author", bib_details.get("bib1_author", []))
    authors_str = ", ".join(authors_list) if isinstance(authors_list, list) and authors_list else "著者不明"
    if isinstance(authors_list, str) and authors_list:
        authors_str = authors_list

    publisher = bib_details.get("publisher", bib_details.get("bib1_publisher", "出版社不明"))
    pubdate = bib_details.get("pubdate", bib_details.get("bib1_pubdate", "出版日不明"))

    return f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"


def load_bib_data(yaml_path):
    """書誌データをYAMLファイルからロードする"""
    global BIB_DATA
    BIB_DATA = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        return False
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        key = 'records' if 'records' in data else 'clusters'
        for _, records in data.get(key, {}).items():
            for record in records:
                record_id = record.get('id') or record.get('record_id')
                if record_id:
                    BIB_DATA[str(record_id)] = record.get('data', record)
        
        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データをロードできませんでした。")
            return False

        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        return True
    except Exception as e:
        print(f"エラー: 書誌データファイルの読み込み中に予期せぬエラー: {e}")
        return False


def select_samples(df, strategy, num_samples, score_col, judgment_col, gt_col):
    """指定された戦略に基づいてデータフレームからサンプルを選択する"""
    print(f"\nサンプリング戦略 '{strategy}' を用いて {num_samples} ペアを選択します...")
    
    if strategy == "lc":  # Least Confidence
        if score_col not in df.columns or judgment_col not in df.columns:
            print(f"エラー: LCサンプリングにはスコア列'{score_col}'と判定列'{judgment_col}'が必要です。")
            return pd.DataFrame()
        
        df_clean = df.dropna(subset=[score_col]).copy()
        df_clean['abs_diff_from_0.5'] = (df_clean[score_col] - 0.5).abs()
        return df_clean.nsmallest(num_samples, 'abs_diff_from_0.5')

    elif strategy == "random_gt":  # Ground Truth based Random
        if gt_col not in df.columns:
            print(f"エラー: random_gtサンプリングには正解列'{gt_col}'が必要です。")
            return pd.DataFrame()

        df_clean = df.dropna(subset=[gt_col]).copy()
        positive_pairs = df_clean[df_clean[gt_col] == True]
        negative_pairs = df_clean[df_clean[gt_col] == False]
        
        n_pos = min(len(positive_pairs), num_samples // 2)
        n_neg = num_samples - n_pos
        
        sampled_pos = positive_pairs.sample(n=n_pos, random_state=42)
        sampled_neg = negative_pairs.sample(n=n_neg, random_state=42)
        return pd.concat([sampled_pos, sampled_neg])

    else:
        print(f"エラー: 未知のサンプリング戦略 '{strategy}'")
        return pd.DataFrame()

def format_for_jsonl(row, judgment_col):
    """ファインチューニング用のJSONL形式にデータを整形する"""
    record_id_1 = row["record_id_1"]
    record_id_2 = row["record_id_2"]

        bib_info_1 = get_record_details_for_prompt(record_id_1)
        bib_info_2 = get_record_details_for_prompt(record_id_2)

    user_prompt = (
            f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\n\n"
            f"書誌情報1:\n{bib_info_1}\n\n"
            f"書誌情報2:\n{bib_info_2}\n\n"
            "これらは同一の文献ですか？\n回答:"
        )

    judgment_val = row[judgment_col]
    if pd.isna(judgment_val): return None
    
    is_similar = str(judgment_val).lower() in ["true", "1", "yes"]
    assistant_response = "はい" if is_similar else "いいえ"

    return {
            "messages": [
            {
                "role": "system", 
                "content": "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。回答は「はい」か「いいえ」のどちらか一方のみで行ってください。"
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="LLM評価結果からファインチューニング用JSONLファイルを生成するスクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-csv", required=True, help="LLM評価結果のCSVファイルパス")
    parser.add_argument("--ground-truth-yaml", required=True, help="書誌情報と正解クラスタのYAMLファイルパス")
    parser.add_argument("--output-jsonl", required=True, help="出力するJSONLファイルのパス")
    parser.add_argument("--judgment-column", default="ground_truth_similar", help="類似判断の真偽値として使用する列名")
    parser.add_argument("--score-column", default="score_after", help="スコアとして使用する列名 (LCサンプリング用)")
    parser.add_argument("--sampling-strategy", choices=["lc", "random_gt"], default="lc", help="ペアのサンプリング戦略")
    parser.add_argument("--num-samples", type=int, default=500, help="サンプリングするペアの総数")

    args = parser.parse_args()

    if not load_bib_data(args.ground_truth_yaml):
        return

    try:
        df = pd.read_csv(args.input_csv, encoding='utf-8-sig')
        print(f"入力CSV {args.input_csv} を読み込みました。合計 {len(df)} ペア。")
    except FileNotFoundError:
        print(f"エラー: 入力CSVファイルが見つかりません: {args.input_csv}")
        return
    
    # 必要な列の型を安全に変換
    if args.score_column in df.columns:
        df[args.score_column] = pd.to_numeric(df[args.score_column], errors='coerce')
    if args.judgment_column in df.columns:
        df[args.judgment_column] = df[args.judgment_column].astype(str)

    sampled_df = select_samples(
        df, args.sampling_strategy, args.num_samples, 
        args.score_column, args.judgment_column, 'ground_truth_similar'
    )

    if sampled_df.empty:
        print("サンプリングされたペアが0件のため、処理を終了します。")
        return

    print(f"{len(sampled_df)} ペアをサンプリングしました。JSONLファイルを作成します...")

    try:
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for _, row in sampled_df.iterrows():
                formatted_data = format_for_jsonl(row, args.judgment_column)
                if formatted_data:
                    f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
        
        print(f"\nファインチューニング用JSONLファイルを {args.output_jsonl} に保存しました。")

    except Exception as e:
        print(f"エラー: JSONLファイルの書き込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
