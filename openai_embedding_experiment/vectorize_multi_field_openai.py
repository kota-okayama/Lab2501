import openai
import os
import time
import argparse
import pickle
import numpy as np
import sys
import json

# data_processing.load_yaml_data をインポートするためにプロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from data_processing.load_yaml_data import load_bibliographic_data


def get_text_from_record(record_data, fields=None):
    """レコードから指定されたフィールドのテキスト表現を生成"""
    if fields is None:
        fields = ["bib1_title", "bib1_author", "bib1_publisher", "bib1_pubdate"]
    
    text_parts = []
    for field in fields:
        if field in record_data and record_data[field]:
            if isinstance(record_data[field], list):
                # リスト内の各要素もNoneや空文字列でないことを確認
                valid_items = [str(item).strip() for item in record_data[field] 
                              if item and str(item).strip()]
                if valid_items:
                    text_parts.append(", ".join(valid_items))
            else:
                text_content = str(record_data[field]).strip()
                if text_content:  # 空文字列でないことを確認
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

    successful_embeddings = []  # (record_id, embedding_vector) のタプルを格納

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
                response = openai.embeddings.create(input=texts_in_batch, 
                                                   model=model)

                api_embeddings = [item.embedding for item in response.data]

                for record_id, embedding_vector in zip(ids_in_batch, 
                                                      api_embeddings):
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


def generate_field_combinations():
    """生成するフィールド組み合わせを定義"""
    return {
        "full": ["bib1_title", "bib1_author", "bib1_publisher", "bib1_pubdate"],
        "title_only": ["bib1_title"],
        "author_only": ["bib1_author"],
        "publisher_only": ["bib1_publisher"],
        "pubdate_only": ["bib1_pubdate"],
        "title_author": ["bib1_title", "bib1_author"],
        "title_publisher": ["bib1_title", "bib1_publisher"],
        "author_publisher": ["bib1_author", "bib1_publisher"]
    }


def process_field_combination(records_list, field_combination, combination_name, args):
    """特定のフィールド組み合わせでエンベディングを生成"""
    print(f"\n=== {combination_name} の処理を開始 ===")
    print(f"使用フィールド: {field_combination}")
    
    # テキスト表現の生成
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

    # エンベディング生成
    embeddings_with_ids = get_embeddings_openai(
        texts_for_api, model=args.openai_model, batch_size=args.api_batch_size
    )

    if not embeddings_with_ids:
        print(f"  警告: {combination_name} で有効なエンベディングが取得できませんでした。")
        return None

    # 結果の保存
    final_record_ids = [item[0] for item in embeddings_with_ids]
    final_embeddings_list = [item[1] for item in embeddings_with_ids]
    embeddings_array = np.array(final_embeddings_list, dtype=np.float32)

    # ファイルパスの生成
    base_dir = os.path.dirname(args.output_embeddings_path)
    embeddings_file = os.path.join(base_dir, f"embeddings_{combination_name}.npy")
    ids_file = os.path.join(base_dir, f"record_ids_{combination_name}.pkl")

    # 保存
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
    parser = argparse.ArgumentParser(description="Generate embeddings for multiple field combinations using OpenAI API.")
    parser.add_argument("--record_yaml_path", type=str, required=True, help="Path to the input YAML file.")
    parser.add_argument(
        "--output_embeddings_path", type=str, required=True, 
        help="Base path for output embeddings files (will be used as template)."
    )
    parser.add_argument(
        "--openai_model", type=str, default="text-embedding-ada-002", 
        help="OpenAI embedding model to use."
    )
    parser.add_argument(
        "--api_batch_size", type=int, default=50, 
        help="Batch size for OpenAI API requests."
    )
    parser.add_argument(
        "--selected_combinations", type=str, default="", 
        help="Comma-separated list of combinations to generate (e.g., 'full,title_only,author_only'). If empty, all combinations will be generated."
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(args.output_embeddings_path), exist_ok=True)

    print(f"レコードの読み込み: {args.record_yaml_path}")
    records_list_from_yaml = load_bibliographic_data(args.record_yaml_path)

    if not records_list_from_yaml:
        print("レコードが読み込めませんでした。処理を終了します。")
        return

    print(f"読み込み成功: {len(records_list_from_yaml)}件のレコード")

    # フィールド組み合わせの取得
    all_combinations = generate_field_combinations()
    
    # 処理対象の組み合わせを決定
    if args.selected_combinations:
        selected_names = [name.strip() for name in args.selected_combinations.split(",")]
        combinations_to_process = {name: all_combinations[name] for name in selected_names if name in all_combinations}
        
        invalid_names = [name for name in selected_names if name not in all_combinations]
        if invalid_names:
            print(f"警告: 無効な組み合わせ名が指定されました: {invalid_names}")
            print(f"利用可能な組み合わせ: {list(all_combinations.keys())}")
    else:
        combinations_to_process = all_combinations

    print(f"\n処理対象の組み合わせ: {list(combinations_to_process.keys())}")

    # 各組み合わせの処理
    results = []
    for combination_name, field_combination in combinations_to_process.items():
        result = process_field_combination(records_list_from_yaml, field_combination, combination_name, args)
        if result:
            results.append(result)

    # 結果サマリーの保存
    summary_file = os.path.join(os.path.dirname(args.output_embeddings_path), "embedding_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== 処理完了 ===")
    print(f"生成されたエンベディング: {len(results)}種類")
    print(f"結果サマリー: {summary_file}")
    
    for result in results:
        print(f"  {result['name']}: {result['record_count']}件 (次元: {result['dimension']})")


if __name__ == "__main__":
    main() 