import argparse
import os
import sys

import pandas as pd
from openai import OpenAI


def load_master_data(filepath):
    """
    レコードIDと内容を含むマスターデータを読み込み、IDをキーとする辞書を返す
    """
    print(f"{filepath} からマスターデータを読み込んでいます...")
    try:
        df = pd.read_csv(filepath)
        # カラム名が 'record_id' と 'content' であることを前提としています
        return pd.Series(df.content.values, index=df.record_id.astype(str)).to_dict()
    except FileNotFoundError:
        print(f"エラー: マスターデータファイルが見つかりません: {filepath}", file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print(f"エラー: マスターデータファイルには 'record_id' と 'content' カラムが必要です。", file=sys.stderr)
        sys.exit(1)

def select_pairs_with_llm(args, client, master_data):
    """ActiveLLMの手法に基づき、LLMを使用してペアを選択する"""
    print("LLMによるペア選択を開始します...")
    try:
        df = pd.read_csv(args.evaluation_details_csv)
    except FileNotFoundError:
        print(f"エラー: 評価詳細ファイルが見つかりません: {args.evaluation_details_csv}", file=sys.stderr)
        sys.exit(1)

    # 1. LLMに提示する候補をランダムにサンプリングする
    if len(df) < args.num_candidates:
        print(f"警告: データセットのサイズ ({len(df)}) が候補数 ({args.num_candidates}) より小さいです。データセット全体を候補とします。")
        candidate_df = df
    else:
        candidate_df = df.sample(n=args.num_candidates, random_state=42)
    
    candidate_pairs = []
    for _, row in candidate_df.iterrows():
        candidate_pairs.append((str(row['record_id_1']), str(row['record_id_2'])))

    # 2. LLMへのプロンプトを生成する
    prompt_header = f"""あなたは、機械学習モデルの訓練に使うためのデータを選択する専門家（アクティブラーナー）です。
これから、2つのレコードのペアのリストを提示します。これらのペアは、2つのレコードが同じエンティティを指しているかどうかを判定するタスクのデータです。
モデルの学習効率が最大になるように、最も有益だと考えられるペアを {args.num_samples} 個選択してください。
選択する際は、多様性、曖昧さ、代表性などを考慮してください。

思考のステップを一つずつ記述してください。

最終的な回答は、選択したペアのインデックスのみをカンマ区切りで出力してください。他の説明は不要です。
例: 1, 5, 12, 28

以下がデータのペアです:
---
"""
    
    prompt_body_parts = []
    for i, (id1, id2) in enumerate(candidate_pairs):
        content1 = master_data.get(id1, "[コンテンツが見つかりません]")
        content2 = master_data.get(id2, "[コンテンツが見つかりません]")
        prompt_body_parts.append(f"{i}: (レコード1: '{content1}', レコード2: '{content2}')")
    
    full_prompt = prompt_header + "\n".join(prompt_body_parts)

    # 3. LLMに問い合わせる
    print(f"{args.num_candidates} 件の候補から {args.num_samples} 件を選択するようLLMに依頼します...")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", # 必要に応じてモデルを変更してください
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        print(f"エラー: OpenAI APIへの問い合わせ中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. LLMの応答からインデックスを抽出する
    try:
        # 応答には思考プロセスが含まれるため、最後の行からインデックスを抽出する
        last_line = response_text.strip().split('\n')[-1]
        selected_indices = [int(i.strip()) for i in last_line.split(',')]
        
        if any(i >= len(candidate_pairs) for i in selected_indices):
            raise ValueError("LLMが範囲外のインデックスを返しました。")
            
        selected_pairs = [candidate_pairs[i] for i in selected_indices]
        print(f"LLMが {len(selected_pairs)} 件のペアを選択しました。")
        return selected_pairs

    except (ValueError, IndexError) as e:
        print(f"エラー: LLMの応答の解析に失敗しました。応答: '{response_text}'", file=sys.stderr)
        print(f"詳細: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="LLMを使用してファインチューニング用のデータペアを選択します。")
    parser.add_argument("--evaluation_details_csv", type=str, required=True, help="評価詳細が記載されたCSVファイルのパス")
    parser.add_argument("--master_data_file", type=str, required=True, help="レコードIDと内容を含むマスターデータCSVのパス")
    parser.add_argument("--num_samples", type=int, required=True, help="LLMに選択させるペアの数")
    parser.add_argument("--num_candidates", type=int, default=200, help="LLMに提示する候補の数")
    parser.add_argument("--output_file", type=str, required=True, help="選択されたペアを保存するCSVファイルのパス")
    args = parser.parse_args()

    # OpenAIクライアントの初期化
    if "OPENAI_API_KEY" not in os.environ:
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)
    client = OpenAI()
    
    # マスターデータの読み込み
    master_data = load_master_data(args.master_data_file)
    
    # LLMによるペア選択の実行
    selected_pairs = select_pairs_with_llm(args, client, master_data)
    
    # 結果の保存
    output_df = pd.DataFrame(selected_pairs, columns=['record_id_1', 'record_id_2'])
    output_df.to_csv(args.output_file, index=False)
    print(f"選択されたペアを {args.output_file} に保存しました。")

if __name__ == "__main__":
    main()