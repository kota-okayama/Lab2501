import openai
import os
import time
import argparse
import pickle
import numpy as np
import sys
import json
import re

# data_processing.load_yaml_data をインポートするためにプロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from data_processing.load_yaml_data import load_yaml_data


def get_text_from_record(record_data, fields):
    """レコードから指定されたフィールドのテキスト表現を生成"""
    text_parts = []
    for field in fields:
        if field in record_data and record_data[field]:
            if isinstance(record_data[field], list):
                valid_items = [str(item).strip() for item in record_data[field] 
                              if item and str(item).strip()]
                if valid_items:
                    text_parts.append(", ".join(valid_items))
            else:
                text_content = str(record_data[field]).strip()
                if text_content:
                    text_parts.append(text_content)

    return " ".join(text_parts) if text_parts else None


def get_embeddings_openai(texts_with_ids, model="text-embedding-ada-002", 
                         api_key=None, batch_size=100, retry_delay=5):
    """OpenAI APIを使用してテキストのエンベディングを取得"""
    if api_key:
        openai.api_key = api_key
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set the "
                           "OPENAI_API_KEY environment variable.")

    successful_embeddings = []

    for i in range(0, len(texts_with_ids), batch_size):
        batch_data = texts_with_ids[i: i + batch_size]
        texts_in_batch = [item[1] for item in batch_data]
        ids_in_batch = [item[0] for item in batch_data]

        if not texts_in_batch:
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                batch_num = i//batch_size + 1
                total_batches = (len(texts_with_ids) - 1) // batch_size + 1
                print(
                    f"  バッチ {batch_num} / {total_batches}: "
                    f"{len(texts_in_batch)}件のテキストをAPIに送信中..."
                )
                response = openai.embeddings.create(input=texts_in_batch, model=model)
                api_embeddings = [item.embedding for item in response.data]
                for record_id, embedding_vector in zip(ids_in_batch, api_embeddings):
                    successful_embeddings.append((record_id, embedding_vector))
                print(f"  バッチ {batch_num} 処理完了。")
                if i + batch_size < len(texts_with_ids):
                    time.sleep(1)
                break
            except Exception as e:
                print(f"  APIエラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
                else:
                    print("  最大リトライ回数に達しました。このバッチの処理をスキップします。")
                    break
    return successful_embeddings


def parse_embedding_combinations(combinations_str, available_fields):
    """組み合わせ指定文字列をパースして辞書に変換する"""
    if not combinations_str:
        combinations_str = "full"

    combinations = {}
    parts = combinations_str.split(';')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part == "full":
            combinations['full'] = available_fields
            continue

        match = re.match(r'\[(.*)\]', part)
        if match:
            # 複数フィールドの組み合わせ: [title,artist]
            fields = [f.strip() for f in match.group(1).split(',')]
            name = "_".join(fields)
        else:
            # 単一フィールド
            fields = [part]
            name = part
        
        # フィールドの妥当性チェック
        invalid_fields = [f for f in fields if f not in available_fields]
        if invalid_fields:
            print(f"エラー: 指定されたフィールドが存在しません: {invalid_fields}")
            print(f"利用可能なフィールド: {available_fields}")
            sys.exit(1)
        
        combinations[name] = fields
        
    return combinations


def process_field_combination(records_list, field_combination, combination_name, output_base_dir, args):
    """特定のフィールド組み合わせでエンベディングを生成"""
    print(f"\n=== {combination_name} の処理を開始 ===")
    print(f"使用フィールド: {field_combination}")
    
    texts_for_api = []
    for record_entry in records_list:
        record_data = record_entry.get("data", {})
        record_id = record_entry.get("record_id")
        if not record_id:
            continue
        text_representation = get_text_from_record(record_data, fields=field_combination)
        if text_representation:
            texts_for_api.append((record_id, text_representation))

    if not texts_for_api:
        print(f"  警告: {combination_name} で有効なテキストが見つかりませんでした。スキップします。")
        return None

    print(f"  {len(texts_for_api)}件のレコードをベクトル化します...")

    embeddings_with_ids = get_embeddings_openai(
        texts_for_api, model=args.openai_model, batch_size=args.api_batch_size
    )

    if not embeddings_with_ids:
        print(f"  警告: {combination_name} で有効なエンベディングが取得できませんでした。")
        return None

    final_record_ids = [item[0] for item in embeddings_with_ids]
    final_embeddings_list = [item[1] for item in embeddings_with_ids]
    embeddings_array = np.array(final_embeddings_list, dtype=np.float32)

    embeddings_file = os.path.join(output_base_dir, f"embeddings_{combination_name}.npy")
    ids_file = os.path.join(output_base_dir, f"record_ids_{combination_name}.pkl")

    np.save(embeddings_file, embeddings_array)
    with open(ids_file, "wb") as f:
        pickle.dump(final_record_ids, f)

    print(f"  {combination_name}: {len(final_record_ids)}件のベクトル保存完了")
    print(f"    エンベディング: {embeddings_file}")
    print(f"    レコードID: {ids_file}")

    return {
        "name": combination_name,
        "fields": field_combination,
        "embeddings_file": embeddings_file,
        "ids_file": ids_file,
        "record_count": len(final_record_ids),
        "dimension": embeddings_array.shape[1] if embeddings_array.ndim > 1 else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for specified field combinations from a YAML file."
    )
    parser.add_argument("--record_yaml_path", type=str, required=True, 
                        help="Path to the input YAML file.")
    parser.add_argument("--output_base_dir", type=str, required=True, 
                        help="Base directory for all output files.")
    parser.add_argument("--openai_model", type=str, default="text-embedding-ada-002", 
                        help="OpenAI embedding model to use.")
    parser.add_argument("--api_batch_size", type=int, default=50, 
                        help="Batch size for OpenAI API requests.")
    parser.add_argument(
        "--embedding_combinations", type=str, default="full", 
        help="Semicolon-separated list of field combinations. "
             "E.g., 'full;title;[title,artist]'. "
             "Defaults to 'full' if not provided."
    )

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    print(f"レコードの読み込み: {args.record_yaml_path}")
    records_list, inf_attr = load_yaml_data(args.record_yaml_path)

    if not records_list:
        print("レコードが読み込めませんでした。処理を終了します。")
        return
    print(f"読み込み成功: {len(records_list)}件のレコード")

    available_fields = list(inf_attr.keys())
    if not available_fields:
        print("エラー: YAMLファイルに 'inf_attr' が見つからないか、空です。")
        print("利用可能なフィールドを特定できませんでした。")
        sys.exit(1)

    combinations_to_process = parse_embedding_combinations(
        args.embedding_combinations, available_fields
    )

    print(f"\n処理対象の組み合わせ: {list(combinations_to_process.keys())}")

    results = []
    for name, fields in combinations_to_process.items():
        result = process_field_combination(
            records_list, fields, name, args.output_base_dir, args
        )
        if result:
            results.append(result)

    summary_file = os.path.join(args.output_base_dir, "embedding_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== 処理完了 ===")
    print(f"生成されたエンベディング: {len(results)}種類")
    print(f"結果サマリー: {summary_file}")
    
    for result in results:
        print(f"  {result['name']}: {result['record_count']}件 (次元: {result['dimension']})")


if __name__ == "__main__":
    main() 