import openai
import os
import time
import argparse
import pickle
import numpy as np
import sys

# data_processing.load_yaml_data をインポートするためにプロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from data_processing.load_yaml_data import load_bibliographic_data


def get_text_from_record(record_data, fields=["bib1_title", "bib1_author", "bib1_publisher", "bib1_pubdate"]):
    text_parts = []
    for field in fields:
        if field in record_data and record_data[field]:
            if isinstance(record_data[field], list):
                # リスト内の各要素もNoneや空文字列でないことを確認
                valid_items = [str(item).strip() for item in record_data[field] if item and str(item).strip()]
                if valid_items:
                    text_parts.append(", ".join(valid_items))
            else:
                text_content = str(record_data[field]).strip()
                if text_content:  # 空文字列でないことを確認
                    text_parts.append(text_content)

    return " ".join(text_parts) if text_parts else None


def get_embeddings_openai(texts_with_ids, model="text-embedding-ada-002", api_key=None, batch_size=100, retry_delay=5):
    if api_key:
        openai.api_key = api_key
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    successful_embeddings = []  # (record_id, embedding_vector) のタプルを格納

    # texts_with_ids は (record_id, text_content) のリストを期待

    for i in range(0, len(texts_with_ids), batch_size):
        batch_data = texts_with_ids[i : i + batch_size]

        # APIに送るテキストと、それに対応するIDを抽出
        # (この時点でテキストがNoneや空のものはtexts_with_idsに含まれていない前提)
        texts_in_batch = [item[1] for item in batch_data]
        ids_in_batch = [item[0] for item in batch_data]

        if not texts_in_batch:  # 万が一空のバッチならスキップ
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"  バッチ {i//batch_size + 1} / { (len(texts_with_ids) -1 ) // batch_size + 1 }: {len(texts_in_batch)}件のテキストをAPIに送信中..."
                )
                response = openai.embeddings.create(input=texts_in_batch, model=model)

                api_embeddings = [item.embedding for item in response.data]

                # 得られた埋め込みとIDをペアにして保存
                for record_id, embedding_vector in zip(ids_in_batch, api_embeddings):
                    successful_embeddings.append((record_id, embedding_vector))

                print(f"  バッチ {i//batch_size + 1} 処理完了。")
                if i + batch_size < len(texts_with_ids):
                    time.sleep(1)
                break
            except Exception as e:
                print(f"  APIエラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
                else:
                    print(f"  最大リトライ回数に達しました。このバッチの処理をスキップします。")
                    # このバッチのIDは結果に含まれなくなる
                    break
    return successful_embeddings


def main():
    parser = argparse.ArgumentParser(description="Vectorize bibliographic records using OpenAI Embedding API.")
    parser.add_argument("--record_yaml_path", type=str, required=True, help="Path to the input YAML file.")
    parser.add_argument(
        "--output_embeddings_path", type=str, required=True, help="Path to save the .npy embeddings file."
    )
    parser.add_argument("--output_ids_path", type=str, required=True, help="Path to save the .pkl record IDs file.")
    parser.add_argument(
        "--text_fields",
        type=str,
        default="bib1_title,bib1_author,bib1_publisher,bib1_pubdate",
        help="Comma-separated list of fields to use for text representation.",
    )
    parser.add_argument(
        "--openai_model", type=str, default="text-embedding-ada-002", help="OpenAI embedding model to use."
    )
    parser.add_argument(
        "--api_batch_size",
        type=int,
        default=50,
        help="Batch size for OpenAI API requests (max 2048 inputs for some models).",
    )  # デフォルトバッチサイズを少し減らした

    args = parser.parse_args()

    fields_to_use = [f.strip() for f in args.text_fields.split(",")]
    print(f"使用するテキストフィールド: {fields_to_use}")

    os.makedirs(os.path.dirname(args.output_embeddings_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_ids_path), exist_ok=True)

    print(f"Loading records from: {args.record_yaml_path}")
    records_list_from_yaml = load_bibliographic_data(args.record_yaml_path)

    if not records_list_from_yaml:
        print("レコードが読み込めませんでした。処理を終了します。")
        return
    print(f"Successfully loaded {len(records_list_from_yaml)} records.")

    # APIに送信する (record_id, text_representation) のリストを作成
    # テキストがNoneや空の場合はこの時点で除外する
    texts_for_openai_api = []
    print("Generating text representations for records and filtering invalid texts...")
    for record_entry in records_list_from_yaml:
        record_data = record_entry.get("data", {})
        record_id = record_entry.get("record_id")
        if not record_id:
            print(
                f"  警告: record_idがないレコードが見つかりました。スキップします: {record_entry[:100]}..."
            )  # 全体表示を避ける
            continue

        text_representation = get_text_from_record(record_data, fields=fields_to_use)

        if text_representation:  # None や空文字列でない場合のみAPI送信用リストに追加
            texts_for_openai_api.append((record_id, text_representation))
        else:
            print(
                f"  情報: レコードID {record_id} は有効なテキスト表現が生成できなかったため、ベクトル化対象外とします。"
            )

    if not texts_for_openai_api:
        print("ベクトル化対象の有効なテキストがありません。処理を終了します。")
        return

    print(f"{len(texts_for_openai_api)}件のレコードについてOpenAI APIでベクトルを取得します...")

    # (record_id, embedding_vector) のリストが返ってくる
    embeddings_with_ids = get_embeddings_openai(
        texts_for_openai_api, model=args.openai_model, batch_size=args.api_batch_size
    )

    if not embeddings_with_ids:
        print("有効な埋め込みベクトルが1つも得られませんでした。処理を終了します。")
        return

    # 分離して保存
    final_record_ids = [item[0] for item in embeddings_with_ids]
    final_embeddings_list = [item[1] for item in embeddings_with_ids]

    embeddings_array = np.array(final_embeddings_list, dtype=np.float32)

    print(
        f"ベクトル化処理完了。 {len(final_record_ids)} 件のベクトル (次元数: {embeddings_array.shape[1] if embeddings_array.ndim > 1 and embeddings_array.shape[0] > 0 else '不明'}) を保存します。"
    )

    try:
        np.save(args.output_embeddings_path, embeddings_array)
        print(f"Embeddings saved to: {args.output_embeddings_path}")

        with open(args.output_ids_path, "wb") as f_ids:
            pickle.dump(final_record_ids, f_ids)
        print(f"Record IDs saved to: {args.output_ids_path}")
    except Exception as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
